"""Per-month aggregates — feeds the TimeScrubber and Money Clock."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from neo4j import Driver

from src.api.deps import get_driver
from src.api.models import TimelinePoint, TimelineResponse

router = APIRouter(prefix="/timeline", tags=["timeline"])

TIMELINE_QUERY = """
MATCH (mo:Month)
OPTIONAL MATCH (t:Transaction)-[:IN_MONTH]->(mo)
RETURN mo.id AS month,
       sum(CASE WHEN t.amount > 0 THEN  t.amount ELSE 0 END) AS income,
       sum(CASE WHEN t.amount < 0 THEN -t.amount ELSE 0 END) AS expense,
       count(t) AS tx
ORDER BY month
"""


@router.get("", response_model=TimelineResponse)
def get_timeline(driver: Driver = Depends(get_driver)) -> TimelineResponse:
    with driver.session() as s:
        rows = s.run(TIMELINE_QUERY).data()
    return TimelineResponse(points=[
        TimelinePoint(
            month=r["month"],
            income=round(r["income"] or 0, 2),
            expense=round(r["expense"] or 0, 2),
            net=round((r["income"] or 0) - (r["expense"] or 0), 2),
            transactions=r["tx"] or 0,
        )
        for r in rows
    ])


# Aggregated by day-of-month — Powers the Money Clock view. Big spikes on
# the 1st (utilities/council tax), 7th (mortgage + CC settlement), 14th
# (mobile/comms), 25th (savings transfer), and last day (salary credit).
DAY_OF_MONTH_QUERY = """
MATCH (t:Transaction)-[:AT]->(:Merchant)-[:IN_CATEGORY]->(c:Category)
WHERE t.amount < 0 AND c.name <> 'System'
RETURN t.date.day AS day, c.name AS category,
       sum(-t.amount) AS spend, count(t) AS tx
ORDER BY day, spend DESC
"""


@router.get("/day_of_month")
def get_day_of_month(driver: Driver = Depends(get_driver)) -> dict:
    with driver.session() as s:
        rows = s.run(DAY_OF_MONTH_QUERY).data()

    # Aggregate into one entry per day, with category breakdown for tooltips.
    by_day: dict[int, dict] = {
        d: {"day": d, "spend": 0.0, "transactions": 0, "by_category": []}
        for d in range(1, 32)
    }
    for r in rows:
        d = by_day[r["day"]]
        d["spend"] += float(r["spend"])
        d["transactions"] += int(r["tx"])
        d["by_category"].append({
            "category": r["category"],
            "spend": round(float(r["spend"]), 2),
            "transactions": int(r["tx"]),
        })
    points = [
        {**v, "spend": round(v["spend"], 2)}
        for v in by_day.values()
        # Drop trailing 29/30/31 if no months have that day.
        if v["spend"] > 0 or v["transactions"] > 0
    ]
    return {"points": points}
