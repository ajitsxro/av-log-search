"""
latency.py

Measures search latency through the running API, using the 20 eval queries.

Part A, structured search (database only): POST /search with each query's
expected filter. One untimed warmup, then RUNS_STRUCTURED timed requests.
Part B, text search (LLM + database): GET /search/text with each query's
text, RUNS_TEXT timed requests.

Each request records the client's total time and the server's own timings
(db_ms, and parse_ms for text search). Reports p50, p95 and max:
p95 means 95% of requests were at least this fast.

The API must be running first:
    python -m uvicorn api.main:app

Run from the repo root with:
    python -m evaluation.latency
"""

import json
import time

import httpx
import numpy as np

QUERIES_PATH = "evaluation/eval_queries.json"
API_URL = "http://127.0.0.1:8000"
RUNS_STRUCTURED = 50
RUNS_TEXT = 10


def print_stats(name, values):
    p50 = np.percentile(values, 50)
    p95 = np.percentile(values, 95)
    print(f"  {name:22} p50 {p50:8.1f} ms   p95 {p95:8.1f} ms   max {max(values):8.1f} ms   (n={len(values)})")


def main():
    with open(QUERIES_PATH) as queries_file:
        queries = json.load(queries_file)

    structured_total = []
    structured_db = []
    db_p95_by_query = []

    text_total = []
    text_parse = []
    text_db = []
    fallbacks = 0
    retried = 0

    with httpx.Client(base_url=API_URL, timeout=60) as client:
        print("part A: structured search (database only)...")
        for item in queries:
            client.post("/search", json=item["expected"])

            query_db = []
            for run in range(RUNS_STRUCTURED):
                start = time.perf_counter()
                response = client.post("/search", json=item["expected"])
                total_ms = (time.perf_counter() - start) * 1000
                if response.status_code != 200:
                    print("request failed:", response.status_code, response.text)
                    response.raise_for_status()

                body = response.json()
                structured_total.append(total_ms)
                structured_db.append(body["db_ms"])
                query_db.append(body["db_ms"])

            db_p95_by_query.append((float(np.percentile(query_db, 95)), item["query"]))

        print("part B: text search (LLM + database)...")
        for item in queries:
            for run in range(RUNS_TEXT):
                start = time.perf_counter()
                response = client.get("/search/text", params={"q": item["query"]})
                total_ms = (time.perf_counter() - start) * 1000
                if response.status_code != 200:
                    print("request failed:", response.status_code, response.text)
                    response.raise_for_status()

                body = response.json()
                text_total.append(total_ms)
                text_parse.append(body["parse_ms"])
                text_db.append(body["db_ms"])

                if body["parser"] != "llm":
                    fallbacks += 1
                elif body["attempts"] > 1:
                    retried += 1

    print()
    print("A. structured search (POST /search)")
    print_stats("database (server)", structured_db)
    print_stats("total (client)", structured_total)

    db_p95_by_query.sort(reverse=True)
    print("  slowest queries by database p95:")
    for p95, query in db_p95_by_query[:3]:
        print(f"    {p95:6.1f} ms   {query}")

    print()
    print("B. text search (GET /search/text)")
    print_stats("LLM parse (server)", text_parse)
    print_stats("database (server)", text_db)
    print_stats("total (client)", text_total)
    llm_share = np.median(text_parse) / np.median(text_total) * 100
    print(f"  LLM share of median total: {llm_share:.0f}%")
    print(f"  fell back to keywords: {fallbacks} of {len(text_total)} | needed a retry: {retried}")


if __name__ == "__main__":
    main()