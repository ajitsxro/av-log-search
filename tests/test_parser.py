"""
test_parser.py

Unit tests for the keyword fallback parser. The LLM path is checked by
hand, since unit tests should not call a paid external API.

Run from the repo root with:
    python -m pytest -v
"""

import pytest

from api.parser import QueryParseError, parse_with_keywords


def test_headline_query():
    search_filter = parse_with_keywords("pedestrian within 10 meters at night")

    assert search_filter.classes == ["person"]
    assert search_filter.max_distance_m == 10
    assert search_filter.time_of_day == "night"


def test_synonyms_map_to_stored_classes():
    search_filter = parse_with_keywords("cyclists and scooters in the rain")

    assert search_filter.classes == ["bicycle", "motorcycle"]
    assert search_filter.weather == "rain"


def test_min_distance_and_count():
    search_filter = parse_with_keywords("at least 3 cars beyond 30 m during the day")

    assert search_filter.classes == ["car"]
    assert search_filter.min_count == 3
    assert search_filter.min_distance_m == 30
    assert search_filter.time_of_day == "day"


def test_no_object_type_raises():
    with pytest.raises(QueryParseError):
        parse_with_keywords("something interesting at night")


def test_rule_violations_raise_parse_error():
    """Traffic lights with a distance filter break a SearchFilter rule."""
    with pytest.raises(QueryParseError):
        parse_with_keywords("traffic lights within 20 m")