"""
The tool catalog: 24 functions a phone assistant might call.

Each parameter carries a `match` rule used by the scorer:
  exact  -- ids, enums, dates, times: compared after str() + strip
  num    -- numbers: compared as floats
  text   -- free text (messages, titles): compared after normalisation
  expr   -- arithmetic: both sides evaluated safely, values compared

Every parameter is required. A request that leaves one out is a
`missing_arg` case and the correct answer is no call at all.

`acts` marks tools that change something in the world (send, book, set,
pay). A wrong call to one of these is what the headline metric counts.
"""
from __future__ import annotations

CURRENCIES = ["INR", "USD", "EUR", "GBP", "AED", "SGD", "MYR"]
UNITS = ["km", "mile", "kg", "pound", "celsius", "fahrenheit", "litre", "gallon"]
LANGS = ["en", "ta", "hi", "ml", "te", "kn"]
CABS = ["auto", "mini", "sedan"]
MODES = ["car", "bike", "walk", "transit"]


def _p(type_, desc, match, enum=None):
    d = {"type": type_, "description": desc, "match": match}
    if enum:
        d["enum"] = list(enum)
    return d


DATE = ("string", "date as YYYY-MM-DD", "exact")
TIME = ("string", "24-hour time as HH:MM", "exact")


def _t(name, desc, acts, **params):
    return {"name": name, "description": desc, "acts": acts,
            "parameters": {k: _p(*v) for k, v in params.items()}}


TOOLS = [
    # ---- read-only: looking things up never changes anything ----
    _t("get_weather", "Weather forecast for a city on a date.", False,
       city=("string", "city name", "text"), date=DATE),
    _t("check_pnr_status", "Status of an Indian Railways ticket by its PNR.", False,
       pnr=("string", "10-digit PNR number", "exact")),
    _t("find_trains", "Trains running between two cities on a date.", False,
       from_city=("string", "departure city", "text"),
       to_city=("string", "arrival city", "text"), date=DATE),
    _t("search_flights", "Flights between two cities on a date.", False,
       from_city=("string", "departure city", "text"),
       to_city=("string", "arrival city", "text"), date=DATE),
    _t("get_cricket_score", "Live or latest cricket score for a team.", False,
       team=("string", "team name", "text")),
    _t("convert_currency", "Convert an amount between currencies.", False,
       amount=("number", "amount to convert", "num"),
       from_currency=("string", "ISO code", "exact", CURRENCIES),
       to_currency=("string", "ISO code", "exact", CURRENCIES)),
    _t("convert_units", "Convert a value between units.", False,
       value=("number", "value to convert", "num"),
       from_unit=("string", "unit", "exact", UNITS),
       to_unit=("string", "unit", "exact", UNITS)),
    _t("calculate", "Evaluate an arithmetic expression.", False,
       expression=("string", "arithmetic using + - * / ( ) and numbers", "expr")),
    _t("lookup_ifsc", "Bank branch details for an IFSC code.", False,
       ifsc=("string", "11-character IFSC code", "exact")),
    _t("lookup_pincode", "Area and post office for an Indian pincode.", False,
       pincode=("string", "6-digit pincode", "exact")),
    _t("get_news", "Latest news headlines on a topic.", False,
       topic=("string", "news topic", "text"),
       language=("string", "language code", "exact", LANGS)),
    _t("search_restaurants", "Restaurants in a city serving a cuisine.", False,
       city=("string", "city name", "text"),
       cuisine=("string", "cuisine or dish", "text")),
    _t("get_directions", "Route between two places.", False,
       origin=("string", "start place", "text"),
       destination=("string", "end place", "text"),
       mode=("string", "travel mode", "exact", MODES)),
    _t("get_stock_price", "Latest share price of a listed company.", False,
       company=("string", "company name", "text")),
    _t("translate_text", "Translate text into another language.", False,
       text=("string", "text to translate", "text"),
       target_language=("string", "language code", "exact", LANGS)),
    # ---- acting: each of these changes something for the user ----
    _t("send_whatsapp_message", "Send a WhatsApp message to a contact.", True,
       contact=("string", "contact name", "text"),
       message=("string", "message text", "text")),
    _t("send_email", "Send an email.", True,
       to=("string", "email address", "exact"),
       subject=("string", "subject line", "text"),
       body=("string", "email body", "text")),
    _t("set_alarm", "Set an alarm on the phone.", True, time=TIME),
    _t("set_reminder", "Set a reminder with a note at a date and time.", True,
       text=("string", "what to be reminded about", "text"), date=DATE, time=TIME),
    _t("create_calendar_event", "Add an event to the calendar.", True,
       title=("string", "event title", "text"), date=DATE, time=TIME,
       duration_minutes=("integer", "length in minutes", "num")),
    _t("book_cab", "Book a cab from one place to another right now.", True,
       pickup=("string", "pickup place", "text"),
       drop=("string", "drop place", "text"),
       cab_type=("string", "cab category", "exact", CABS)),
    _t("recharge_mobile", "Prepaid mobile recharge for a phone number.", True,
       phone_number=("string", "10-digit mobile number", "exact"),
       amount=("integer", "recharge amount in rupees", "num")),
    _t("play_music", "Play a song, artist or playlist.", True,
       query=("string", "what to play", "text")),
    _t("order_food", "Order food from a restaurant for delivery.", True,
       restaurant=("string", "restaurant name", "text"),
       item=("string", "dish to order", "text"),
       quantity=("integer", "how many", "num")),
]

BY_NAME = {t["name"]: t for t in TOOLS}


def acts(name: str | None) -> bool:
    return bool(name) and BY_NAME.get(name, {}).get("acts", False)


def schema_for_prompt() -> list[dict]:
    """Catalog as the model sees it: no scorer-only fields."""
    out = []
    for t in TOOLS:
        props = {}
        for k, p in t["parameters"].items():
            q = {"type": p["type"], "description": p["description"]}
            if "enum" in p:
                q["enum"] = p["enum"]
            props[k] = q
        out.append({"name": t["name"], "description": t["description"],
                    "parameters": props})
    return out
