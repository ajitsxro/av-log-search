"""
precision.py

Measures search quality on 20 natural-language queries
(evaluation/eval_queries.json), each with a hand-written expected filter.

For every query:
1. Parse accuracy: does the LLM produce exactly the expected filter?
2. Search precision@10: run the expected filter and check each returned
   frame against nuScenes labels.
3. End-to-end precision@10: the same check on the LLM's filter.

A frame is correct when its labels contain at least min_count objects of
the asked classes that meet the distance conditions (distance to the
object's nearest surface, as in the distance validation).

Visibility: "strict" counts only labels at least 40% visible (nuScenes
level >= 2); "lenient" counts every label in the front camera. nuScenes
visibility is measured across all 6 cameras, not the front camera alone.

time_of_day and weather are not checked: they come from the human-written
scene descriptions the search itself uses, so there is no independent truth.

Precision@10 = correct frames / frames returned (at most 10).

Run from the repo root with:
    python -m evaluation.precision
"""

import json

import psycopg
from nuscenes.nuscenes import NuScenes

from api.filters import SearchFilter, filter_to_sql
from api.parser import parse_query
from config import DATABASE_URL, DATAROOT, NUSCENES_VERSION
from evaluation.labels import labeled_boxes_in_image

QUERIES_PATH = "evaluation/eval_queries.json"
STRICT_VISIBILITY = 2
LENIENT_VISIBILITY = 1


def label_meets_filter(label, search_filter, min_visibility):
    if label["class_name"] not in search_filter.classes:
        return False
    if label["visibility"] < min_visibility:
        return False

    distance = label["surface_distance"]
    if search_filter.max_distance_m is not None and distance > search_filter.max_distance_m:
        return False
    if search_filter.min_distance_m is not None and distance < search_filter.min_distance_m:
        return False

    return True


def frame_is_correct(labeled, search_filter, min_visibility):
    count = 0
    for label in labeled:
        if label_meets_filter(label, search_filter, min_visibility):
            count += 1
    return count >= search_filter.min_count


def run_and_judge(conn, nusc, search_filter):
    sql, params = filter_to_sql(search_filter)
    rows = conn.execute(sql, params).fetchall()

    strict_correct = 0
    lenient_correct = 0
    scenes = set()
    for row in rows:
        frame_id = row[0]
        scenes.add(row[1])

        camera_token = conn.execute(
            "SELECT camera_token FROM frames WHERE frame_id = %s", (frame_id,)
        ).fetchone()[0]
        labeled = labeled_boxes_in_image(nusc, camera_token)

        if frame_is_correct(labeled, search_filter, STRICT_VISIBILITY):
            strict_correct += 1
        if frame_is_correct(labeled, search_filter, LENIENT_VISIBILITY):
            lenient_correct += 1

    returned = len(rows)
    if returned == 0:
        return {"returned": 0, "strict": None, "lenient": None, "scenes": 0}

    return {
        "returned": returned,
        "strict": strict_correct / returned,
        "lenient": lenient_correct / returned,
        "scenes": len(scenes),
    }


def format_score(value):
    if value is None:
        return "  -  "
    return f"{value:.2f}"


def average(values):
    kept = []
    for value in values:
        if value is not None:
            kept.append(value)
    if len(kept) == 0:
        return None
    return sum(kept) / len(kept)


def main():
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=DATAROOT, verbose=False)

    with open(QUERIES_PATH) as queries_file:
        queries = json.load(queries_file)

    strict_scores = []
    lenient_scores = []
    end_to_end_scores = []
    parse_correct = 0
    parse_misses = []

    print("query                                      n  strict  lenient  scenes  parse  e2e")
    with psycopg.connect(DATABASE_URL) as conn:
        for item in queries:
            expected_filter = SearchFilter.model_validate(item["expected"])
            search_result = run_and_judge(conn, nusc, expected_filter)

            parsed = parse_query(item["query"])
            llm_filter = parsed["filter"]
            parse_ok = llm_filter == expected_filter

            if parse_ok:
                parse_correct += 1
                end_to_end_result = search_result
            else:
                parse_misses.append((item["query"], parsed["parser"], llm_filter))
                end_to_end_result = run_and_judge(conn, nusc, llm_filter)

            strict_scores.append(search_result["strict"])
            lenient_scores.append(search_result["lenient"])
            end_to_end_scores.append(end_to_end_result["strict"])

            print(
                f"{item['query'][:40]:40} {search_result['returned']:3}  "
                f"{format_score(search_result['strict']):6}  {format_score(search_result['lenient']):7}  "
                f"{search_result['scenes']:6}  {'yes' if parse_ok else 'NO':5}  "
                f"{format_score(end_to_end_result['strict'])}"
            )

    print()
    print("mean search precision@10 (strict): ", format_score(average(strict_scores)))
    print("mean search precision@10 (lenient):", format_score(average(lenient_scores)))
    print("mean end-to-end precision@10 (strict):", format_score(average(end_to_end_scores)))
    print("parse accuracy:", parse_correct, "of", len(queries))

    if len(parse_misses) > 0:
        print()
        print("parse misses:")
        for query, parser_name, llm_filter in parse_misses:
            print("  ", query, "|", parser_name, "|", llm_filter.model_dump(exclude_defaults=True))


if __name__ == "__main__":
    main()