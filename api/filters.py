"""
filters.py

the search filter: the structured form of a natural-language query, and
the code that turns it into a parameterized SQL query.

the LLM only produces SearchFilter JSON; it never writes SQL. Pydantic
rejects anything outside the allowed shape (unknown classes, invented
fields, impossible distance ranges) before it reaches the database.

a frame matches when it has at least min_count detections that pass every
detection condition (class, confidence, distance), and the frame itself
passes time_of_day and weather. results are ranked by the most confident
matching detection, so the top results are the ones the system is surest of.
detections with unknown distance never match a distance condition.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

ObjectClass = Literal["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]


class SearchFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[ObjectClass] = Field(min_length=1)
    max_distance_m: float | None = Field(default=None, gt=0, le=100)
    min_distance_m: float | None = Field(default=None, ge=0, le=100)
    time_of_day: Literal["day", "night"] | None = None
    weather: Literal["clear", "rain"] | None = None
    min_count: int = Field(default=1, ge=1, le=50)
    min_confidence: float = Field(default=0.3, ge=0.1, le=1.0)
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def check_distance_rules(self):
        has_distance = self.max_distance_m is not None or self.min_distance_m is not None

        if self.min_distance_m is not None and self.max_distance_m is not None:
            if self.min_distance_m >= self.max_distance_m:
                raise ValueError("min_distance_m must be less than max_distance_m")

        if has_distance and "traffic light" in self.classes:
            raise ValueError("distance filters are not supported for traffic lights (unverified)")

        return self


def filter_to_sql(search_filter):
    conditions = ["d.class_name = ANY(%s)", "d.confidence >= %s"]
    params = [search_filter.classes, search_filter.min_confidence]

    if search_filter.max_distance_m is not None:
        conditions.append("d.distance_m <= %s")
        params.append(search_filter.max_distance_m)

    if search_filter.min_distance_m is not None:
        conditions.append("d.distance_m >= %s")
        params.append(search_filter.min_distance_m)

    if search_filter.time_of_day is not None:
        conditions.append("f.time_of_day = %s")
        params.append(search_filter.time_of_day)

    if search_filter.weather is not None:
        conditions.append("f.weather = %s")
        params.append(search_filter.weather)

    where_sql = " AND ".join(conditions)

    sql = f"""
        SELECT f.frame_id, f.scene_name, f.image_path, f.time_of_day, f.weather,
               COUNT(*) AS match_count,
               MAX(d.confidence) AS best_confidence,
               MIN(d.distance_m) AS nearest_m
        FROM detections d
        JOIN frames f ON f.frame_id = d.frame_id
        WHERE {where_sql}
        GROUP BY f.frame_id
        HAVING COUNT(*) >= %s
        ORDER BY best_confidence DESC, f.frame_id
        LIMIT %s
    """
    params.append(search_filter.min_count)
    params.append(search_filter.limit)

    return sql, params