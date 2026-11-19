"""
main.py

The search API.

Endpoints:
    POST /search        body: SearchFilter JSON -> matching frames, best first
    GET  /frame/{id}    one frame and its detections (confidence >= 0.3 by default)
    GET  /image/{id}    the frame's JPEG
    GET /search/text  ?q=<words> -> parsed filter + matching frames

FastAPI validates every request against the Pydantic models, so a bad
filter returns a 422 error with the reason and never reaches the database.
A connection pool keeps a few database connections open so each request
does not pay to open a new one. /search reports its database time (db_ms).

Run from the repo root with:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

import os
import time
from contextlib import asynccontextmanager
from api.parser import QueryParseError, parse_query
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from api.filters import SearchFilter, filter_to_sql
from config import AWS_REGION, DATABASE_URL, DATAROOT, IMAGE_SOURCE, S3_BUCKET

pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=False)
PRESIGNED_URL_SECONDS = 3600
s3_client = None

def get_s3_client():
    global s3_client
    if s3_client is None:
        import boto3
        from botocore.config import Config
        s3_client = boto3.client("s3", region_name=AWS_REGION, config=Config(signature_version="s3v4"))
    return s3_client


@asynccontextmanager
async def lifespan(app):
    pool.open()
    yield
    pool.close()


app = FastAPI(title="av-log-search", lifespan=lifespan)


class SearchResult(BaseModel):
    frame_id: int
    scene_name: str
    time_of_day: str
    weather: str
    match_count: int
    best_confidence: float
    nearest_m: float | None
    image_url: str


class SearchResponse(BaseModel):
    filter: SearchFilter
    db_ms: float
    results: list[SearchResult]


class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    distance_m: float | None


class FrameDetail(BaseModel):
    frame_id: int
    scene_name: str
    time_of_day: str
    weather: str
    image_url: str
    detections: list[Detection]

class TextSearchResponse(BaseModel):
    query: str
    parser: str
    attempts: int
    fallback_reason: str | None
    filter: SearchFilter
    parse_ms: float
    db_ms: float
    results: list[SearchResult]

def run_search(search_filter):
    sql, params = filter_to_sql(search_filter)

    start = time.perf_counter()
    with pool.connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    db_ms = (time.perf_counter() - start) * 1000

    results = []
    for row in rows:
        frame_id, scene_name, image_path, time_of_day, weather, match_count, best_confidence, nearest_m = row
        results.append(SearchResult(
            frame_id=frame_id,
            scene_name=scene_name,
            time_of_day=time_of_day,
            weather=weather,
            match_count=match_count,
            best_confidence=round(best_confidence, 3),
            nearest_m=None if nearest_m is None else round(nearest_m, 1),
            image_url=f"/image/{frame_id}",
        ))

    return results, round(db_ms, 2)


@app.post("/search", response_model=SearchResponse)
def search(search_filter: SearchFilter):
    results, db_ms = run_search(search_filter)
    return SearchResponse(filter=search_filter, db_ms=db_ms, results=results)

@app.get("/search/text", response_model=TextSearchResponse)
def search_text(q: str = Query(min_length=1, max_length=300)):
    start = time.perf_counter()
    try:
        parsed = parse_query(q)
    except QueryParseError as error:
        raise HTTPException(status_code=422, detail=str(error))
    parse_ms = (time.perf_counter() - start) * 1000

    results, db_ms = run_search(parsed["filter"])

    return TextSearchResponse(
        query=q,
        parser=parsed["parser"],
        attempts=parsed["attempts"],
        fallback_reason=parsed["fallback_reason"],
        filter=parsed["filter"],
        parse_ms=round(parse_ms, 2),
        db_ms=db_ms,
        results=results,
    )


@app.get("/frame/{frame_id}", response_model=FrameDetail)
def get_frame(frame_id: int, min_confidence: float = Query(default=0.3, ge=0.1, le=1.0)):
    with pool.connection() as conn:
        frame = conn.execute(
            "SELECT frame_id, scene_name, time_of_day, weather FROM frames WHERE frame_id = %s",
            (frame_id,),
        ).fetchone()
        if frame is None:
            raise HTTPException(status_code=404, detail="frame not found")

        rows = conn.execute(
            "SELECT class_name, confidence, x1, y1, x2, y2, distance_m FROM detections "
            "WHERE frame_id = %s AND confidence >= %s ORDER BY confidence DESC",
            (frame_id, min_confidence),
        ).fetchall()

    detections = []
    for class_name, confidence, x1, y1, x2, y2, distance_m in rows:
        detections.append(Detection(
            class_name=class_name,
            confidence=round(confidence, 3),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            distance_m=None if distance_m is None else round(distance_m, 1),
        ))

    return FrameDetail(
        frame_id=frame[0],
        scene_name=frame[1],
        time_of_day=frame[2],
        weather=frame[3],
        image_url=f"/image/{frame_id}",
        detections=detections,
    )


@app.get("/image/{frame_id}")
def get_image(frame_id: int):
    with pool.connection() as conn:
        row = conn.execute("SELECT image_path FROM frames WHERE frame_id = %s", (frame_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="frame not found")

    image_path = row[0]

    if IMAGE_SOURCE == "s3":
        url = get_s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": image_path},
            ExpiresIn=PRESIGNED_URL_SECONDS,
        )
        return RedirectResponse(url)

    return FileResponse(os.path.join(DATAROOT, image_path), media_type="image/jpeg")

@app.get("/")
def home():
    return FileResponse("web/index.html")