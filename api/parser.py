"""
parser.py

turns a natural-language query into a validated SearchFilter.

primary path: claude haiku with structured outputs returns a ParsedQuery
(only the fields a user's words control). it is then validated as a
SearchFilter; if that fails, the validation error is sent back to the model
and it retries, up to MAX_ATTEMPTS.

fallback path: a keyword parser with no network calls, used when the LLM is
unavailable (no key, API error) or never returns a valid filter.

the LLM never sets min_confidence or limit; those stay system defaults.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from api.filters import ObjectClass, SearchFilter

LLM_MODEL = "claude-haiku-5-5"
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You convert a search query about self-driving car camera footage into a filter.

Fields:
- classes: object types to find. Allowed: person, bicycle, car, motorcycle, bus, truck, traffic light.
  Map synonyms: pedestrian/people/walker -> person; cyclist/bike -> bicycle;
  motorbike/scooter -> motorcycle; vehicle -> car; lorry -> truck.
- max_distance_m: for "within X m", "closer than X m", "less than X meters". Convert feet to meters.
- min_distance_m: for "farther than X m", "beyond X m", "more than X meters away".
- time_of_day: "night" or "day", only if the query says so ("dark" means night).
- weather: "rain" for rain/raining/wet; "clear" only if the query says clear or dry.
- min_count: the number of objects required, only if the query gives one ("at least 3 cars" -> 3).

"The car", "our car", "the ego vehicle" or "us" mean the camera car itself, not an object to find.

Set a field to null if the query does not mention it. Do not guess."""

CLASS_KEYWORDS = {
    "person": ["pedestrian", "pedestrians", "person", "people", "walker", "walkers"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes", "cyclist", "cyclists"],
    "car": ["car", "cars", "vehicle", "vehicles"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "scooter", "scooters"],
    "bus": ["bus", "buses"],
    "truck": ["truck", "trucks", "lorry", "lorries"],
    "traffic light": ["traffic light", "traffic lights"],
}
UNIT = r"\s*(?:m|meter|meters|metre|metres)\b"
MAX_DISTANCE_PATTERN = r"(?:within|under|less than|closer than|below)\s+(\d+(?:\.\d+)?)" + UNIT
MIN_DISTANCE_PATTERN = r"(?:beyond|over|more than|farther than|further than)\s+(\d+(?:\.\d+)?)" + UNIT
COUNT_PATTERN = r"(?:at least\s+)?(\d+)\s+(?:or more\s+)?(?:pedestrians|people|persons|cars|vehicles|trucks|buses|bikes|bicycles|cyclists|motorcycles|motorbikes|scooters)"


class QueryParseError(Exception):
    pass


class ParsedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[ObjectClass]
    max_distance_m: float | None
    min_distance_m: float | None
    time_of_day: Literal["day", "night"] | None
    weather: Literal["clear", "rain"] | None
    min_count: int | None


llm_client = None


def get_llm_client():
    global llm_client
    if llm_client is None:
        from anthropic import Anthropic
        llm_client = Anthropic()
    return llm_client


def parse_with_llm(text):
    client = get_llm_client()
    messages = [{"role": "user", "content": text}]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.messages.parse(
            model=LLM_MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_format=ParsedQuery,
        )
        parsed = response.parsed_output
        if parsed is None:
            continue

        try:
            search_filter = SearchFilter.model_validate(parsed.model_dump(exclude_none=True))
            return search_filter, attempt
        except ValidationError as error:
            messages.append({"role": "assistant", "content": parsed.model_dump_json()})
            messages.append({
                "role": "user",
                "content": "That filter failed validation:\n" + str(error) + "\nReturn a corrected filter.",
            })

    raise QueryParseError(f"no valid filter after {MAX_ATTEMPTS} attempts")


def parse_with_keywords(text):
    lowered = text.lower()
    words = re.findall(r"[a-z]+", lowered)

    classes = []
    for class_name, keywords in CLASS_KEYWORDS.items():
        for keyword in keywords:
            is_phrase = " " in keyword
            if (is_phrase and keyword in lowered) or (not is_phrase and keyword in words):
                classes.append(class_name)
                break

    if len(classes) == 0:
        raise QueryParseError("no object type found in query")

    values = {"classes": classes}

    if "night" in words or "dark" in words:
        values["time_of_day"] = "night"
    elif "day" in words or "daytime" in words:
        values["time_of_day"] = "day"

    if "rain" in words or "raining" in words or "rainy" in words or "wet" in words:
        values["weather"] = "rain"

    max_match = re.search(MAX_DISTANCE_PATTERN, lowered)
    if max_match:
        values["max_distance_m"] = float(max_match.group(1))

    min_match = re.search(MIN_DISTANCE_PATTERN, lowered)
    if min_match:
        values["min_distance_m"] = float(min_match.group(1))

    count_match = re.search(COUNT_PATTERN, lowered)
    if count_match:
        values["min_count"] = int(count_match.group(1))

    try:
        return SearchFilter.model_validate(values)
    except ValidationError as error:
        raise QueryParseError(str(error))


def parse_query(text):
    try:
        search_filter, attempts = parse_with_llm(text)
        return {"filter": search_filter, "parser": "llm", "attempts": attempts, "fallback_reason": None}
    except Exception as error:
        fallback_reason = type(error).__name__ + ": " + str(error)

    search_filter = parse_with_keywords(text)
    return {"filter": search_filter, "parser": "keyword", "attempts": 0, "fallback_reason": fallback_reason}