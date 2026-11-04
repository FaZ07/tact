"""
Dataset generator. Fully enumerated: no random module anywhere. Every choice
is a SHA-256 of (split, template, fill index, slot tag), so the same code
writes the same bytes on every machine.

What keeps the test set honest:
  * held-out phrasings  -- each tool's last two templates appear ONLY in test
  * held-out values     -- test cities, contacts, songs, dishes ... never in train
  * moving "today"      -- every example has its own date, so "tomorrow" has
                           to be computed, not memorised
  * cases built to lose -- near_miss (sounds like a tool, none exists) and
                           missing_arg (the right tool, a detail left out);
                           the correct answer to both is no call

Categories: plain, ta_mixed, near_miss, missing_arg, no_tool.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

from .prompt import WEEKDAYS
from .tools import BY_NAME

N_FILLS = {"train": 20, "test": 8}
REF0 = dt.date(2026, 1, 1)
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class Pick:
    def __init__(self, *seed):
        self.seed = "|".join(map(str, seed))

    def n(self, tag: str, k: int) -> int:
        return h(self.seed, tag) % k

    def of(self, tag: str, seq):
        return seq[self.n(tag, len(seq))]


# (train values, test-only values)
POOLS = {
    "city": (["Chennai", "Madurai", "Coimbatore", "Tirunelveli", "Bengaluru",
              "Mumbai", "Delhi", "Hyderabad", "Kochi", "Pune", "Trichy", "Salem"],
             ["Kolkata", "Mysuru", "Thoothukudi", "Vellore", "Jaipur", "Nagercoil"]),
    "team": (["India", "Australia", "England", "Pakistan", "CSK", "MI", "RCB",
              "KKR", "South Africa", "New Zealand", "Sri Lanka", "Tamil Nadu"],
             ["Gujarat Titans", "Afghanistan", "Bangladesh", "West Indies",
              "Rajasthan Royals"]),
    "who": (["Amma", "Appa", "Ravi", "Priya", "Karthik", "Meena", "Arjun",
             "Divya", "Anna", "Selvi"],
            ["Akka", "Senthil", "Lakshmi", "Faisal", "Thatha"]),
    "msg": (["I will be late by 10 minutes", "dinner is ready",
             "call me when you reach", "the meeting moved to 4",
             "I reached home safely", "bring milk on the way"],
            ["the train is delayed", "happy anniversary", "keep the gate open"]),
    "mail": (["ravi", "priya", "karthik", "meena", "arjun", "divya"],
             ["lakshmi", "faisal", "senthil"]),
    "subj": (["Leave request", "Project update", "Invoice for October",
              "Meeting notes", "Fee receipt"],
             ["Viva schedule", "Rent reminder"]),
    "body": (["please find the details attached", "I will be on leave tomorrow",
              "the report is ready for review", "thanks for your help"],
             ["the payment has been made", "can we meet next week"]),
    "note": (["pay the electricity bill", "call the doctor", "take my tablets",
              "renew the bike insurance", "submit the assignment",
              "water the plants"],
             ["book the gas cylinder", "collect the passport"]),
    "title": (["Team standup", "Dentist appointment", "Parent teacher meeting",
               "Gym session", "Project review", "Driving class"],
              ["Viva rehearsal", "Bank visit"]),
    "place": (["Central Station", "the airport", "Marina Beach",
               "Meenakshi Temple", "my office", "T Nagar", "Gandhipuram",
               "the bus stand", "Anna Nagar", "Phoenix Mall"],
              ["Koyambedu", "Tidel Park", "Brookefields Mall", "Ukkadam",
               "Velachery"]),
    "song": (["Ilaiyaraaja hits", "AR Rahman songs", "Anirudh latest",
              "Vaathi Coming", "lo-fi study music", "Kannalane", "Rowdy Baby"],
             ["Yuvan BGMs", "SPB melodies", "Arabic Kuthu"]),
    "rest": (["A2B", "Saravana Bhavan", "Junior Kuppanna", "Thalappakatti",
              "Murugan Idli Shop", "Sangeetha"],
             ["Anjappar", "Buhari"]),
    "item": (["chicken biryani", "masala dosa", "idli", "parotta",
              "mutton chukka", "filter coffee", "veg meals"],
             ["ghee roast", "chicken 65"]),
    "cuisine": (["biryani", "dosa", "chettinad", "chinese", "north indian",
                 "parotta", "pizza", "seafood"],
                ["idiyappam", "mandi", "kerala meals"]),
    "topic": (["cricket", "elections", "stock market", "technology", "budget",
               "ISRO", "movies", "education"],
              ["monsoon", "startups", "football", "farming"]),
    "co": (["Infosys", "TCS", "Reliance", "HDFC Bank", "Tata Motors", "Wipro",
            "ITC", "Asian Paints"],
           ["Titan", "Maruti Suzuki", "Bharti Airtel", "Larsen and Toubro"]),
    "phrase": (["good morning", "where is the bus stand", "thank you very much",
                "I will come tomorrow", "how much does this cost",
                "please call me back", "the food was delicious"],
               ["happy birthday", "the shop is closed today"]),
}

LANG_NAMES = {"en": "English", "ta": "Tamil", "hi": "Hindi", "ml": "Malayalam",
              "te": "Telugu", "kn": "Kannada"}
CUR_WORDS = {"INR": ["rupees", "INR"], "USD": ["dollars", "USD", "US dollars"],
             "EUR": ["euros", "EUR"], "GBP": ["British pounds", "GBP"],
             "AED": ["dirhams", "AED"], "SGD": ["Singapore dollars", "SGD"],
             "MYR": ["ringgit", "MYR"]}
UNIT_PAIRS = [("km", "mile", [1, 5, 10, 21, 42, 100, 2.5]),
              ("kg", "pound", [1, 2, 5, 10, 25, 70, 0.5]),
              ("celsius", "fahrenheit", [0, 25, 30, 37, 40, 100]),
              ("litre", "gallon", [1, 5, 10, 20, 50, 3.5])]
UNIT_WORDS = {"km": ["km", "kilometres"], "mile": ["miles"],
              "kg": ["kg", "kilograms"], "pound": ["pounds", "lbs"],
              "celsius": ["celsius", "degrees Celsius"],
              "fahrenheit": ["fahrenheit", "degrees Fahrenheit"],
              "litre": ["litres", "liters"], "gallon": ["gallons"]}
MODE_WORDS = {"car": ["by car", "driving"], "bike": ["by bike", "on my bike"],
              "walk": ["walking", "on foot"],
              "transit": ["by bus", "by public transport"]}
CAB_WORDS = {"auto": ["an auto"], "mini": ["a mini cab", "a mini"],
             "sedan": ["a sedan"]}
CAB_TA = {"auto": "auto", "mini": "mini cab", "sedan": "sedan"}
DURATIONS = {30: ["for 30 minutes", "for half an hour"], 45: ["for 45 minutes"],
             60: ["for an hour", "for 1 hour"],
             90: ["for 90 minutes", "for an hour and a half"],
             120: ["for 2 hours", "for two hours"]}
QTY_WORDS = {1: ["one", "1"], 2: ["two", "2"], 3: ["three", "3"], 4: ["4", "four"]}
BANKS = ["SBIN", "HDFC", "ICIC", "IOBA", "CNRB", "KKBK", "UTIB", "IDIB"]
RECHARGES = [199, 239, 299, 349, 479, 666, 719]
MONEY = [10, 50, 100, 250, 500, 1000, 1500, 2500, 10000]


def pool(split: str, name: str) -> list:
    return POOLS[name][0 if split == "train" else 1]


def two(p: Pick, split: str, name: str) -> tuple:
    seq = pool(split, name)
    a = p.of(name, seq)
    rest = [x for x in seq if x != a]
    return a, p.of(name + "2", rest)


def fmt_num(x) -> str:
    return str(x)


# ---------------------------------------------------------------- slots
def next_weekday(today: dt.date, wd: int) -> dt.date:
    """Prompt rule 4: the next such weekday after today, never today."""
    return today + dt.timedelta((wd - today.weekday()) % 7 or 7)


def next_month_day(today: dt.date, m: int, d: int) -> dt.date:
    """Prompt rule 4: the next m/d on or after today."""
    target = dt.date(today.year, m, d)
    return target if target >= today else dt.date(today.year + 1, m, d)


def date_slot(p: Pick, today: dt.date, tamil: bool = False) -> tuple[str, str]:
    kinds = (["today", "tomorrow", "day_after"] if tamil else
             ["today", "tomorrow", "day_after", "weekday", "dmy", "mdy"])
    kind = p.of("dkind", kinds)
    if kind == "today":
        return ("innaikku" if tamil else "today"), today.isoformat()
    if kind == "tomorrow":
        return ("naalaikku" if tamil else "tomorrow"), (today + dt.timedelta(1)).isoformat()
    if kind == "day_after":
        return (("naalai marunaal" if tamil else "the day after tomorrow"),
                (today + dt.timedelta(2)).isoformat())
    if kind == "weekday":
        wd = p.n("wd", 7)
        return f"on {WEEKDAYS[wd]}", next_weekday(today, wd).isoformat()
    m, d = p.n("mon", 12) + 1, p.n("day", 28) + 1
    surf = f"on {d} {MONTHS[m - 1]}" if kind == "dmy" else f"on {MONTHS[m - 1]} {d}"
    return surf, next_month_day(today, m, d).isoformat()


def time_slot(p: Pick) -> tuple[str, str]:
    hh, mm = 5 + p.n("hh", 18), [0, 15, 30, 45][p.n("mm", 4)]
    gold = f"{hh:02d}:{mm:02d}"
    form = p.n("tform", 3)
    if form == 2:
        return gold, gold
    if hh == 12 and mm == 0 and form == 0:
        return "noon", gold
    h12, ap = hh % 12 or 12, ("am" if hh < 12 else "pm")
    return (f"{h12} {ap}" if mm == 0 else f"{h12}:{mm:02d} {ap}"), gold


def time_ta(p: Pick) -> tuple[str, str]:
    band = p.n("band", 3)
    if band == 0:
        hr = 5 + p.n("hr", 7)
        return f"kaalai {hr} mani", f"{hr:02d}:00"
    if band == 1:
        hr = 4 + p.n("hr", 3)
        return f"saayangaalam {hr} mani", f"{hr + 12:02d}:00"
    hr = 8 + p.n("hr", 4)
    return f"raathiri {hr} mani", f"{hr + 12:02d}:00"


# --------------------------------------------------------------- frames
# frame(p, split, today) -> (surfaces for the template, gold arguments)
def f_weather(p, s, today, ta=False):
    city = p.of("city", pool(s, "city"))
    ds, dg = date_slot(p, today, ta)
    return {"city": city, "date": ds}, {"city": city, "date": dg}


def f_pnr(p, s, today):
    pnr = f"{4 + p.n('pnr1', 5)}{p.n('pnr', 10 ** 9):09d}"
    return {"pnr": pnr}, {"pnr": pnr}


def f_route(p, s, today):
    a, b = two(p, s, "city")
    ds, dg = date_slot(p, today)
    return ({"city": a, "city2": b, "date": ds},
            {"from_city": a, "to_city": b, "date": dg})


def f_team(p, s, today):
    t = p.of("team", pool(s, "team"))
    return {"team": t}, {"team": t}


def f_currency(p, s, today):
    codes = list(CUR_WORDS)
    c1 = p.of("cur", codes)
    c2 = p.of("cur2", [c for c in codes if c != c1])
    amt = p.of("amt", MONEY)
    return ({"amt": fmt_num(amt), "cur": p.of("cw", CUR_WORDS[c1]),
             "cur2": p.of("cw2", CUR_WORDS[c2])},
            {"amount": amt, "from_currency": c1, "to_currency": c2})


def f_units(p, s, today):
    u1, u2, vals = p.of("pair", UNIT_PAIRS)
    if p.n("dir", 2):
        u1, u2 = u2, u1
    v = p.of("val", vals)
    return ({"val": fmt_num(v), "u1": p.of("uw", UNIT_WORDS[u1]),
             "u2": p.of("uw2", UNIT_WORDS[u2])},
            {"value": v, "from_unit": u1, "to_unit": u2})


def f_calc(p, s, today):
    a, b, c = 2 + p.n("a", 98), 2 + p.n("b", 48), 2 + p.n("c", 8)
    kind = p.n("ck", 6)
    if kind == 0:
        surf, expr = f"{a} times {b}", f"{a}*{b}"
    elif kind == 1:
        surf, expr = f"{a * 13} plus {b * 7}", f"{a * 13}+{b * 7}"
    elif kind == 2:
        surf, expr = f"{a * 11} minus {b}", f"{a * 11}-{b}"
    elif kind == 3:
        pct, n = p.of("pct", [5, 10, 12, 15, 18, 25]), a * 100
        surf, expr = f"{pct} percent of {n}", f"{pct}/100*{n}"
    elif kind == 4:
        surf, expr = f"{a * c} divided by {c}", f"{a * c}/{c}"
    else:
        surf, expr = f"({a} + {b}) * {c}", f"({a}+{b})*{c}"
    return {"calc": surf}, {"expression": expr}


def f_ifsc(p, s, today):
    code = f"{p.of('bank', BANKS)}0{p.n('ifsc', 10 ** 6):06d}"
    return {"ifsc": code}, {"ifsc": code}


def f_pin(p, s, today):
    pin = str(600000 + p.n("pin", 50000))
    return {"pin": pin}, {"pincode": pin}


def f_news(p, s, today):
    topic, lang = p.of("topic", pool(s, "topic")), p.of("lang", list(LANG_NAMES))
    return ({"topic": topic, "lang": "in " + LANG_NAMES[lang]},
            {"topic": topic, "language": lang})


def f_food_search(p, s, today):
    city, cu = p.of("city", pool(s, "city")), p.of("cuisine", pool(s, "cuisine"))
    return {"city": city, "cuisine": cu}, {"city": city, "cuisine": cu}


def f_directions(p, s, today):
    a, b = two(p, s, "place")
    mode = p.of("mode", list(MODE_WORDS))
    return ({"a": a, "b": b, "mode": p.of("mw", MODE_WORDS[mode])},
            {"origin": a, "destination": b, "mode": mode})


def f_stock(p, s, today):
    co = p.of("co", pool(s, "co"))
    return {"co": co}, {"company": co}


def f_translate(p, s, today):
    txt, lang = p.of("phrase", pool(s, "phrase")), p.of("lang", list(LANG_NAMES))
    return ({"txt": txt, "lang": LANG_NAMES[lang]},
            {"text": txt, "target_language": lang})


def f_wa(p, s, today):
    who, msg = p.of("who", pool(s, "who")), p.of("msg", pool(s, "msg"))
    return {"who": who, "msg": msg}, {"contact": who, "message": msg}


def f_email(p, s, today):
    to = p.of("mail", pool(s, "mail")) + "@example.com"
    subj, body = p.of("subj", pool(s, "subj")), p.of("body", pool(s, "body"))
    return ({"to": to, "subj": subj, "body": body},
            {"to": to, "subject": subj, "body": body})


def f_alarm(p, s, today, ta=False):
    ts, tg = time_ta(p) if ta else time_slot(p)
    return {"time": ts}, {"time": tg}


def f_reminder(p, s, today, ta=False):
    note = p.of("note", pool(s, "note"))
    ds, dg = date_slot(p, today, ta)
    ts, tg = time_ta(p) if ta else time_slot(p)
    return ({"note": note, "date": ds, "time": ts},
            {"text": note, "date": dg, "time": tg})


def f_event(p, s, today):
    title = p.of("title", pool(s, "title"))
    ds, dg = date_slot(p, today)
    ts, tg = time_slot(p)
    mins = p.of("dur", list(DURATIONS))
    return ({"title": title, "date": ds, "time": ts,
             "dur": p.of("dw", DURATIONS[mins])},
            {"title": title, "date": dg, "time": tg, "duration_minutes": mins})


def f_cab(p, s, today, ta=False):
    a, b = two(p, s, "place")
    cab = p.of("cab", list(CAB_WORDS))
    word = CAB_TA[cab] if ta else p.of("cw", CAB_WORDS[cab])
    return {"a": a, "b": b, "cab": word}, {"pickup": a, "drop": b, "cab_type": cab}


def f_recharge(p, s, today):
    phone, amt = f"9{p.n('ph', 10 ** 9):09d}", p.of("rc", RECHARGES)
    return ({"phone": phone, "amt": str(amt)},
            {"phone_number": phone, "amount": amt})


def f_music(p, s, today):
    song = p.of("song", pool(s, "song"))
    return {"song": song}, {"query": song}


def f_order(p, s, today):
    rest, item = p.of("rest", pool(s, "rest")), p.of("item", pool(s, "item"))
    q = p.of("qty", list(QTY_WORDS))
    return ({"rest": rest, "item": item, "qty": p.of("qw", QTY_WORDS[q])},
            {"restaurant": rest, "item": item, "quantity": q})


def f_all(p, s, today):
    """Every surface at once, for the no-call templates."""
    surf: dict = {}
    for fr in (f_weather, f_wa, f_recharge, f_order, f_reminder, f_cab,
               f_pnr, f_email, f_team):
        surf.update(fr(p, s, today)[0])
    return surf, {}


def ta(fr):
    return lambda p, s, today: fr(p, s, today, ta=True)


# ------------------------------------------------------------ templates
# (tool or None, category, frame, train texts, test-only texts)
SPEC = [
    ("get_weather", "plain", f_weather,
     ["What's the weather in {city} {date}?", "Will it rain in {city} {date}?",
      "weather forecast {city} {date}", "Should I carry an umbrella in {city} {date}?",
      "How hot will {city} be {date}?"],
     ["{city} weather {date} please",
      "I'm travelling to {city} {date}, how's the weather looking?"]),
    ("check_pnr_status", "plain", f_pnr,
     ["Check PNR {pnr}", "What's the status of my train ticket {pnr}?",
      "PNR status for {pnr}", "Is my ticket with PNR {pnr} confirmed?",
      "pnr {pnr} status pls"],
     ["Has my waitlisted ticket {pnr} got confirmed yet?",
      "Can you look up booking {pnr} on IRCTC?"]),
    ("find_trains", "plain", f_route,
     ["Trains from {city} to {city2} {date}",
      "Which trains go from {city} to {city2} {date}?",
      "I need a train {city} to {city2} {date}",
      "Show trains between {city} and {city2} {date}",
      "Any trains leaving {city} for {city2} {date}?"],
     ["I want to go by rail from {city} to {city2} {date}, what are my options?",
      "List the express trains {city} to {city2} {date}"]),
    ("search_flights", "plain", f_route,
     ["Flights from {city} to {city2} {date}",
      "Find me a flight {city} to {city2} {date}",
      "Are there flights from {city} to {city2} {date}?",
      "Cheapest flight {city} to {city2} {date}",
      "I want to fly from {city} to {city2} {date}"],
     ["Search air tickets {city} to {city2} {date}",
      "What planes are going from {city} to {city2} {date}?"]),
    ("get_cricket_score", "plain", f_team,
     ["What's the {team} score?", "{team} match score",
      "How is {team} doing in the match?", "Live score of {team}",
      "Score update for {team} please"],
     ["Give me the cricket update for {team}",
      "How many runs has {team} made so far?"]),
    ("convert_currency", "plain", f_currency,
     ["Convert {amt} {cur} to {cur2}", "How much is {amt} {cur} in {cur2}?",
      "{amt} {cur} to {cur2}", "What are {amt} {cur} worth in {cur2}?",
      "Change {amt} {cur} into {cur2}"],
     ["My cousin sent {amt} {cur}, how many {cur2} is that?",
      "Exchange value of {amt} {cur} in {cur2} please"]),
    ("convert_units", "plain", f_units,
     ["Convert {val} {u1} to {u2}", "How many {u2} is {val} {u1}?",
      "{val} {u1} in {u2}", "What is {val} {u1} in {u2}?",
      "Change {val} {u1} into {u2}"],
     ["The recipe says {val} {u1}, what's that in {u2}?",
      "I need {val} {u1} expressed in {u2}"]),
    ("calculate", "plain", f_calc,
     ["What is {calc}?", "Calculate {calc}", "{calc} = ?", "Work out {calc}",
      "Compute {calc} please"],
     ["Quick maths, {calc}?", "Can you tell me {calc}?"]),
    ("lookup_ifsc", "plain", f_ifsc,
     ["Which branch is IFSC {ifsc}?", "IFSC {ifsc} details",
      "Find bank branch for {ifsc}", "Look up IFSC code {ifsc}",
      "Where is the bank with IFSC {ifsc}?"],
     ["My cheque has {ifsc} printed on it, which branch is that?",
      "Verify this IFSC for me: {ifsc}"]),
    ("lookup_pincode", "plain", f_pin,
     ["Which area is pincode {pin}?", "Pincode {pin} belongs to which post office?",
      "Look up PIN {pin}", "Where is pincode {pin}?", "{pin} pincode details"],
     ["My parcel says {pin}, which locality is that?",
      "Find the post office for postal code {pin}"]),
    ("get_news", "plain", f_news,
     ["Latest {topic} news {lang}", "Show me {topic} headlines {lang}",
      "Any news on {topic} {lang}?", "Read me {topic} news {lang}",
      "Top stories about {topic} {lang}"],
     ["Catch me up on {topic}, {lang} please",
      "Give me today's {topic} updates {lang}"]),
    ("search_restaurants", "plain", f_food_search,
     ["Find {cuisine} restaurants in {city}", "Where can I eat {cuisine} in {city}?",
      "Good {cuisine} places in {city}", "Suggest a {cuisine} restaurant in {city}",
      "Best {cuisine} in {city}"],
     ["I'm in {city} and craving {cuisine}, any spots?",
      "Hotels serving {cuisine} in {city}"]),
    ("get_directions", "plain", f_directions,
     ["How do I get from {a} to {b} {mode}?", "Directions from {a} to {b} {mode}",
      "Route {a} to {b} {mode}", "Show me the way from {a} to {b} {mode}",
      "Navigate from {a} to {b} {mode}"],
     ["What's the best way {mode} from {a} to {b}?",
      "I'm at {a}, guide me to {b} {mode}"]),
    ("get_stock_price", "plain", f_stock,
     ["What's the share price of {co}?", "{co} stock price",
      "How is {co} stock doing today?", "Current price of {co} shares",
      "Check {co} share value"],
     ["Is {co} up or down on the market right now?",
      "Tell me what one {co} share costs"]),
    ("translate_text", "plain", f_translate,
     ["Translate '{txt}' to {lang}", "How do you say '{txt}' in {lang}?",
      "Say '{txt}' in {lang}", "{lang} translation of '{txt}'",
      "Convert '{txt}' into {lang}"],
     ["My grandmother only reads {lang}, translate this for her: '{txt}'",
      "Put '{txt}' in {lang} please"]),
    ("send_whatsapp_message", "plain", f_wa,
     ["Send {who} a WhatsApp saying {msg}", "WhatsApp {who}: {msg}",
      "Tell {who} on WhatsApp: {msg}", "Text {who} on WhatsApp '{msg}'",
      "Drop a WhatsApp to {who} saying {msg}"],
     ["Ping {who} on WhatsApp with this: {msg}",
      "Let {who} know on WhatsApp: {msg}"]),
    ("send_email", "plain", f_email,
     ["Email {to} with subject '{subj}' saying {body}",
      "Send a mail to {to}, subject {subj}, body: {body}",
      "Write to {to}: subject '{subj}', message '{body}'",
      "Compose an email to {to} titled '{subj}' that says {body}",
      "Mail {to} subject '{subj}' and write {body}"],
     ["Shoot an email over to {to} - subject: {subj} - text: {body}",
      "Please email {to}; put '{subj}' as the subject and '{body}' as the body"]),
    ("set_alarm", "plain", f_alarm,
     ["Set an alarm for {time}", "Wake me up at {time}", "Alarm at {time}",
      "Put an alarm at {time}", "I need to wake up at {time}, set an alarm"],
     ["Can you make my phone ring at {time} tomorrow?", "Buzz me at {time}"]),
    ("set_reminder", "plain", f_reminder,
     ["Remind me to {note} {date} at {time}",
      "Set a reminder {date} at {time} to {note}",
      "Reminder: {note}, {date} at {time}",
      "Don't let me forget to {note} {date} at {time}",
      "At {time} {date}, remind me to {note}"],
     ["Nudge me {date} at {time} about this: {note}",
      "I have to {note} {date} at {time}, remind me"]),
    ("create_calendar_event", "plain", f_event,
     ["Add '{title}' to my calendar {date} at {time} {dur}",
      "Schedule {title} {date} at {time} {dur}",
      "Create a calendar event {title} {date} at {time} {dur}",
      "Put {title} in my calendar {date} at {time}, {dur}",
      "Block my calendar {date} at {time} {dur} for {title}"],
     ["I have {title} {date} at {time}, it runs {dur} - add it",
      "Calendar: {title}, {date}, {time}, {dur}"]),
    ("book_cab", "plain", f_cab,
     ["Book {cab} from {a} to {b}", "Get me {cab} from {a} to {b}",
      "I need {cab} from {a} to {b} now", "Call {cab} to take me from {a} to {b}",
      "Arrange {cab} from {a} to {b}"],
     ["Can you get {cab} to pick me up at {a} and drop me at {b}?",
      "Ride needed: {cab}, {a} to {b}"]),
    ("recharge_mobile", "plain", f_recharge,
     ["Recharge {phone} for Rs {amt}", "Do a Rs {amt} recharge on {phone}",
      "Top up {phone} with {amt} rupees", "Recharge my number {phone} with {amt}",
      "Put Rs {amt} recharge on {phone}"],
     ["Amma's prepaid {phone} needs a {amt} rupee recharge",
      "Please add a Rs {amt} pack to {phone}"]),
    ("play_music", "plain", f_music,
     ["Play {song}", "Put on some {song}", "I want to listen to {song}",
      "Start playing {song}", "Play {song} on my phone"],
     ["Could you queue up {song}?", "Music please: {song}"]),
    ("order_food", "plain", f_order,
     ["Order {qty} {item} from {rest}", "Get me {qty} {item} from {rest}",
      "I want {qty} {item} delivered from {rest}",
      "Place an order at {rest} for {qty} {item}", "Order from {rest}: {qty} {item}"],
     ["Can {rest} send {qty} {item} to my place?",
      "We're hungry - {qty} {item} from {rest} please"]),
    # ---- Tamil-English mixed (romanised). Machine-written: needs a native read.
    ("get_weather", "ta_mixed", ta(f_weather),
     ["{city} la {date} weather eppadi irukkum?"], ["{date} {city} la mazhai varuma?"]),
    ("set_alarm", "ta_mixed", ta(f_alarm),
     ["{time} ku alarm vei"], ["{time} ku ennai ezhuppi vidu"]),
    ("send_whatsapp_message", "ta_mixed", f_wa,
     ["{who} ku WhatsApp la '{msg}' nu anuppu"], ["{who} kitta WhatsApp la sollu: {msg}"]),
    ("set_reminder", "ta_mixed", ta(f_reminder),
     ["{date} {time} ku {note} nu remind pannu"],
     ["{note} - {date} {time} ku nyabagam paduthu"]),
    ("book_cab", "ta_mixed", ta(f_cab),
     ["{a} la irundhu {b} ku {cab} book pannu"], ["{a} to {b} {cab} venum, book panniru"]),
    ("play_music", "ta_mixed", f_music,
     ["{song} podu"], ["{song} kekkanum, play pannu"]),
    ("get_cricket_score", "ta_mixed", f_team,
     ["{team} score enna?"], ["{team} match eppadi poguthu?"]),
    ("search_restaurants", "ta_mixed", f_food_search,
     ["{city} la nalla {cuisine} kadai sollu"], ["{city} la {cuisine} enga kedaikum?"]),
    # ---- no call is correct. near_miss: sounds like a tool, none exists.
    (None, "near_miss", f_all,
     ["Book a hotel room in {city} {date}", "Transfer Rs {amt} to {who} on GPay",
      "What's my bank account balance?", "Call {who}", "Cancel my train ticket {pnr}",
      "Delete all my WhatsApp messages from {who}",
      "Book a movie ticket in {city} {date}", "Turn on the bedroom light",
      "Pay my electricity bill", "Order a new phone charger on Amazon"],
     ["Book a bus ticket from {city} {date}", "Send Rs {amt} to {who}'s bank account",
      "Video call {who}", "Cancel the cab to {b}", "Block {who} on WhatsApp",
      "Book a doctor's appointment {date} at {time}", "Lock my phone"]),
    # missing_arg: the right tool exists, a required detail was never given.
    (None, "missing_arg", f_all,
     ["Set an alarm", "Send a WhatsApp to {who}", "Book {cab} to {b}",
      "Remind me to {note}", "Recharge my phone for Rs {amt}",
      "Translate this into Tamil", "What's the weather {date}?",
      "Order {qty} {item}", "Set a reminder {date} at {time}",
      "Email {to} about it"],
     ["Check my PNR status", "Send an email to {to}",
      "Put a meeting in my calendar {date}", "Play something",
      "Find trains to {city} {date}", "Recharge {phone}", "Message {who} on WhatsApp"]),
    # no_tool: ordinary chat.
    (None, "no_tool", f_all,
     ["Tell me a joke", "Who wrote Thirukkural?", "What is the capital of Australia?",
      "Explain what photosynthesis is", "How are you today?",
      "Write a short poem about rain", "What does GDP mean?",
      "Who won the 2011 cricket world cup?", "Give me a tip to sleep better",
      "Summarise the plot of Ponniyin Selvan", "Thanks, that's all"],
     ["Why is the sky blue?", "Recommend a good Tamil novel",
      "Who is the chief minister of Kerala?",
      "Can you help me prepare for an interview?", "Good night!"]),
]


def templates() -> list[dict]:
    out, count = [], {}
    for tool, cat, frame, train, test in SPEC:
        for split, texts in (("train", train), ("test", test)):
            for text in texts:
                key = f"{tool or cat}/{cat}"
                count[key] = count.get(key, 0) + 1
                out.append({"id": f"{key}/{count[key]}", "tool": tool,
                            "category": cat, "split": split, "frame": frame,
                            "text": text})
    return out


def gold_call(tool: str | None, args: dict) -> dict:
    if tool is None:
        return {"tool": None}
    return {"tool": tool,
            "arguments": {k: args[k] for k in BY_NAME[tool]["parameters"]}}


def build(split: str) -> list[dict]:
    rows, seen = [], set()
    for t in templates():
        if t["split"] != split:
            continue
        reps = N_FILLS[split] if "{" in t["text"] else 1
        for i in range(reps):
            p = Pick(split, t["id"], i)
            today = REF0 + dt.timedelta(p.n("today", 730))
            surf, args = t["frame"](p, split, today)
            query = t["text"].format(**surf)
            key = (query, today)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"id": f"{split}-{len(rows):05d}", "template": t["id"],
                         "category": t["category"], "today": today.isoformat(),
                         "query": query, "gold": gold_call(t["tool"], args)})
    return rows


def dumps(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows)


def write(data_dir: str) -> dict:
    os.makedirs(data_dir, exist_ok=True)
    counts = {}
    for split in ("train", "test"):
        rows = build(split)
        with open(os.path.join(data_dir, f"{split}.jsonl"), "w",
                  encoding="ascii", newline="\n") as f:
            f.write(dumps(rows))
        counts[split] = len(rows)
    return counts


def load(path: str) -> list[dict]:
    with open(path, encoding="ascii") as f:
        return [json.loads(line) for line in f if line.strip()]
