"""
test_filters.py

Unit tests for the search filter: the validation rules LLM output must pass,
and the filter-to-SQL translation (fixed SQL text, values as parameters).

Run from the repo root with:
    python -m pytest -v
"""

import pytest
from pydantic import ValidationError

from api.filters import SearchFilter, filter_to_sql


def test_minimal_filter_gets_defaults():
    search_filter = SearchFilter(classes=["person"])

    assert search_filter.min_confidence == 0.3
    assert search_filter.min_count == 1
    assert search_filter.limit == 10
    assert search_filter.max_distance_m is None


def test_headline_query_parses():
    """'Pedestrian within 10 m at night' as the LLM should produce it."""
    search_filter = SearchFilter.model_validate(
        {"classes": ["person"], "max_distance_m": 10, "time_of_day": "night", "min_count": 1}
    )

    assert search_filter.classes == ["person"]
    assert search_filter.max_distance_m == 10
    assert search_filter.time_of_day == "night"


def test_rejects_unknown_class():
    """'pedestrian' is not a stored class; it must be mapped to 'person'."""
    with pytest.raises(ValidationError):
        SearchFilter(classes=["pedestrian"])


def test_rejects_invented_fields():
    """A field the system does not know must fail, not be silently ignored."""
    with pytest.raises(ValidationError):
        SearchFilter.model_validate({"classes": ["car"], "color": "red"})


def test_rejects_empty_classes():
    with pytest.raises(ValidationError):
        SearchFilter(classes=[])


def test_rejects_negative_distance():
    with pytest.raises(ValidationError):
        SearchFilter(classes=["car"], max_distance_m=-5)


def test_rejects_min_distance_not_below_max():
    with pytest.raises(ValidationError):
        SearchFilter(classes=["car"], min_distance_m=20, max_distance_m=10)


def test_rejects_distance_filter_on_traffic_lights():
    with pytest.raises(ValidationError):
        SearchFilter(classes=["traffic light"], max_distance_m=20)


def test_sql_for_headline_query():
    sql, params = filter_to_sql(
        SearchFilter(classes=["person"], max_distance_m=10, time_of_day="night")
    )

    assert "d.distance_m <= %s" in sql
    assert "f.time_of_day = %s" in sql
    assert "f.weather = %s" not in sql
    assert params == [["person"], 0.3, 10, "night", 1, 10]


def test_sql_leaves_out_unset_filters():
    sql, params = filter_to_sql(SearchFilter(classes=["car"]))

    assert "d.distance_m <= %s" not in sql
    assert "d.distance_m >= %s" not in sql
    assert "f.time_of_day = %s" not in sql
    assert params == [["car"], 0.3, 1, 10]


def test_values_never_appear_in_sql_text():
    """The SQL text has no quoted strings at all; every value is a parameter."""
    sql, params = filter_to_sql(
        SearchFilter(classes=["truck", "bus"], time_of_day="day", weather="rain", min_count=2)
    )

    assert "'" not in sql
    assert params == [["truck", "bus"], 0.3, "day", "rain", 2, 10]