"""Generate dummy Halifax savings + credit card statements as Markdown.

Design
------
* Recurring direct debits, salary, and credit-card settlements are produced
  deterministically in Python so the running balance stays correct.
* Discretionary spend (Tesco baskets, Just Eat orders, fuel at Costco, etc.)
  is generated per-month by ``gpt-5.4-mini`` via a strict JSON schema, so the
  data feels lived-in rather than templated.
* Output Markdown mirrors the section structure of the real samples under
  ``data/statements_masked`` — same headings, same column layout — but with a
  cleaner table that downstream parsers can ingest reliably.
"""
from __future__ import annotations

import calendar
import json
import os
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Persona — fictional customer used across every generated statement.
# ---------------------------------------------------------------------------

PERSONA = {
    "name": "Mr Tony Stark",
    "address_line": "14 Avengers Close",
    "town": "Watford",
    "county": "Hertfordshire",
    # Synthetic UK postcode — uses a non-existent district number so it cannot
    # geocode to a real Royal Mail delivery point.
    "postcode": "WD99 9ZZ",
    # Synthetic sort code — standard test value, never routes a real payment.
    "sort_code": "11-22-33",
    "savings_account": "12345678",
    # Card number is masked except for the last four digits (also synthetic).
    "card_number": "5286 83** **** 1588",
    "credit_limit": 11200.00,
    "employer": "STARK INDUSTRIES",
    "monthly_net_salary": 7312.91,
    "salary_payday_offset_from_month_end": 0,  # last day of month
    "credit_card_last4": "1588",
    # Ofcom-style fictional numbers; the 01632/07700 ranges are reserved
    # for fiction so they can never connect to a real subscriber.
    "support_phone": "01632 960000",
    "card_services_phone": "01632 960123",
}

# Opening balance on 01 January 2025 — chosen to roughly match the trend in
# the real samples (account hovers around £4k-£8k mid-month).
OPENING_SAVINGS_BALANCE = 6_240.00

# ---------------------------------------------------------------------------
# Recurring direct debits & standing orders out of the savings account.
# ``day`` is the day-of-month the payment is taken; clamped to month length.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recurring:
    day: int
    description: str
    amount: float
    tx_type: str = "DD"


RECURRING_OUTFLOWS: list[Recurring] = [
    # 1st of the month — utilities & insurance bundle
    Recurring(1, "WATFORD BOROUGH COUNCIL", 342.89),
    Recurring(1, "E.ON NEXT LTD", 187.42),
    Recurring(1, "AFFINITY WATER", 65.00),
    Recurring(1, "HOME INSURANCE LBIS", 36.94),
    Recurring(1, "SCOTTISH WIDOWS PENSION", 57.42),
    Recurring(1, "SCOTTISH WIDOWS LIFE", 30.99),
    Recurring(1, "TV LICENCE QBP1", 14.17),
    Recurring(1, "SWALE HEATING LTD", 14.17),
    Recurring(1, "WWF UK", 3.00),
    Recurring(1, "CREATION CONSUMER FIN", 90.94),
    # Mid-month — comms & motor insurance
    Recurring(4, "ADMIRAL MOTOR INSURANCE", 90.20),
    Recurring(7, "NATIONWIDE BS MORTGAGE", 1_485.00),
    Recurring(10, "ADMIRAL HOME INSURANCE", 92.44),
    Recurring(14, "VODAFONE LTD DEVICE", 31.00),
    Recurring(15, "VODAFONE LTD", 86.27),
    Recurring(15, "EE LIMITED MOBILE", 28.50),
    Recurring(28, "VIRGIN MEDIA PYMTS", 56.50),
    # Standing orders
    Recurring(25, "PEPPER STARK SAVINGS", 800.00, tx_type="SO"),
]

# Smaller variable extras that always appear but vary slightly.
SAVE_THE_CHANGE_PER_MONTH = (8, 14)  # min, max number of tiny rounding txns

# ---------------------------------------------------------------------------
# Merchant catalogue handed to gpt-5.4-mini so the variable spend stays
# grounded in real UK brands and matches the persona's apparent lifestyle.
# ---------------------------------------------------------------------------

DEBIT_MERCHANTS = [
    "TESCO STORES 3372 WATFORD",
    "TESCO STORES 6753 WATFORD",
    "SAINSBURYS S/MKTS WATFORD",
    "COSTCO WHOLESALE #WATFORD",
    "COSTCO PFS - WATFORD",
    "MARKS & SPENCER PLC",
    "WM MORRISONS STORE 586",
    "B&M 828 WATFORD",
    "BOOTS WATFORD",
    "SHELL CROXLEY 191",
    "PAYPAL *JUSTEATCOUK",
    "UBER EATS",
    "NETFLIX.COM",
    "PAYPAL *DISNEYPLUS",
    "SPOTIFY UK",
    "ALLI BHAVAN WATFORD",
    "PAYPAL *PYPL PAYIN",
    "AMAZON.CO.UK",
]

CREDIT_CARD_MERCHANTS = [
    "TESCO STORES 3372",
    "SAINSBURY'S S/MKT WATFORD",
    "UBER *TRIP HELP.UBER.COM",
    "PAYPAL *JUSTEATCOUK",
    "HELLOFRESH UK LONDON",
    "AMAZON.CO.UK *R684K9ZJ4",
    "AMZNMktplace*R66EF9ZC4",
    "APPLE.COM/BILL CORK IRL",
    "GOOGLE *YouTubePremium",
    "OPENAI *CHATGPT SUBSCR",
    "AWS EMEA aws.amazon.co LUX",
    "TFL TRAVEL CH TFL.GOV.UK/CP",
    "GITHUB, INC. SAN FRANCISCO CA",
    "TUTORFUL* SHEFFIELD",
    "PARENTPAY E-COM R BRIDGWATER",
    "VISION EXPRESS WATFORD",
    "UNIQLO WATFORD",
    "FOOT LOCKER INC 4237",
    "WELCOME BREAK NEWPORT PAGNE",
    "B & Q 1245 WATFORD",
    "JAMAICA BLUE WATFORD",
]

# ---------------------------------------------------------------------------
# Domain objects.
# ---------------------------------------------------------------------------

@dataclass
class Tx:
    """Single ledger entry on the savings statement."""
    date: date
    description: str
    tx_type: str
    money_in: float | None = None
    money_out: float | None = None
    balance: float = 0.0


@dataclass
class CardTx:
    """Single credit-card transaction line."""
    tx_date: date
    posted_date: date
    description: str
    amount: float
    is_credit: bool = False  # True for refunds / payments received


@dataclass
class MonthlySpend:
    """LLM payload: one month of discretionary spend on both products."""
    debit: list[dict] = field(default_factory=list)   # savings-account spend
    credit: list[dict] = field(default_factory=list)  # credit-card spend


# ---------------------------------------------------------------------------
# Date helpers.
# ---------------------------------------------------------------------------

def iter_months(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield ``(month_start, month_end)`` inclusive for every month in range."""
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        end_day = calendar.monthrange(cur.year, cur.month)[1]
        month_end = date(cur.year, cur.month, end_day)
        # On the final month, clip to user-requested end date.
        yield cur, min(month_end, end)
        cur = _next_month(cur)


def _next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _clamp_day(year: int, month: int, day: int) -> int:
    return min(day, calendar.monthrange(year, month)[1])


def _shift_to_weekday(d: date) -> date:
    """Push weekend direct debits to the following Monday (UK BACS behaviour)."""
    while d.weekday() >= 5:
        d = d + timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# LLM hook.
# ---------------------------------------------------------------------------

SPEND_SCHEMA_PROMPT = """You produce realistic, dummy month-of-spend data for a
fictional UK customer ("Tony Stark") living in Watford. Return ONLY valid JSON.

Schema:
{
  "debit": [   // Spend posting to the Halifax Reward Current Account
     {"day": 1-31, "description": "<UPPERCASE MERCHANT>", "amount": <float £>,
      "type": "DEB" | "FPO" | "BP"}
  ],
  "credit": [  // Spend posting to the Halifax Clarity Mastercard
     {"day": 1-31, "description": "<UPPERCASE MERCHANT>", "amount": <float £>}
  ]
}

Rules:
* 14-22 debit-card entries: groceries, fuel, takeaways, small subscriptions.
* 35-55 credit-card entries: family shopping, kids' tutoring, transport,
  online subscriptions, weekend outings.
* Use the supplied merchant catalogue. Mix uppercase chain names exactly as
  they appear; you can append a city/branch suffix.
* Amounts: groceries £6-£135, fuel £40-£90, takeaways £8-£42, subscriptions
  £0.99-£24.99, household £4-£60. Round to 2dp.
* Distribute days across the whole month; weekends are heavier for leisure.
* No duplicates with the same (day, description, amount).
* Reflect seasonality: more spend in Dec (gifts) and Aug (holidays), tighter
  in Jan and Feb.
"""


def _llm_monthly_spend(month_label: str, seed: int) -> MonthlySpend:
    """Call gpt-5.4-mini for one month's variable spend.

    Falls back to a deterministic synthesizer when no API key is present, so
    the generator works offline (useful in CI and for first-run smoke tests).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").removeprefix("openai:")

    if not api_key:
        return _offline_monthly_spend(seed)

    try:
        from openai import OpenAI
    except ImportError:
        return _offline_monthly_spend(seed)

    base_url = os.getenv("OPENAI_BASE_URL") or None
    client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    user_msg = (
        f"Month: {month_label}\n"
        f"Debit merchant catalogue: {DEBIT_MERCHANTS}\n"
        f"Credit merchant catalogue: {CREDIT_CARD_MERCHANTS}\n"
        f"Random seed (use to vary): {seed}\n"
        "Produce the JSON now."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SPEND_SCHEMA_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.9,
        )
        payload = json.loads(resp.choices[0].message.content)
        return MonthlySpend(
            debit=list(payload.get("debit", [])),
            credit=list(payload.get("credit", [])),
        )
    except Exception as exc:  # pragma: no cover — fall back rather than crash
        print(f"[warn] LLM call failed for {month_label}: {exc}; using offline mode")
        return _offline_monthly_spend(seed)


def _offline_monthly_spend(seed: int) -> MonthlySpend:
    """Deterministic fallback when the LLM is unavailable."""
    rng = random.Random(seed)
    debit = []
    for _ in range(rng.randint(14, 22)):
        merchant = rng.choice(DEBIT_MERCHANTS)
        debit.append({
            "day": rng.randint(2, 28),
            "description": merchant,
            "amount": round(rng.uniform(2.50, 145.00), 2),
            "type": "DEB" if "PAYPAL" not in merchant else "DEB",
        })
    credit = []
    for _ in range(rng.randint(35, 55)):
        merchant = rng.choice(CREDIT_CARD_MERCHANTS)
        credit.append({
            "day": rng.randint(1, 28),
            "description": merchant,
            "amount": round(rng.uniform(1.50, 120.00), 2),
        })
    return MonthlySpend(debit=debit, credit=credit)


# ---------------------------------------------------------------------------
# Savings statement assembly.
# ---------------------------------------------------------------------------

def _build_savings_ledger(
    month_start: date,
    month_end: date,
    opening_balance: float,
    spend: MonthlySpend,
    credit_card_settlement: float,
    rng: random.Random,
) -> list[Tx]:
    """Combine recurring + variable + salary into a sorted, balance-tracked ledger."""
    entries: list[Tx] = []

    # Recurring direct debits, weekend-shifted.
    for rec in RECURRING_OUTFLOWS:
        day = _clamp_day(month_start.year, month_start.month, rec.day)
        d = _shift_to_weekday(date(month_start.year, month_start.month, day))
        if d > month_end:
            continue
        entries.append(Tx(d, rec.description, rec.tx_type, money_out=rec.amount))

    # Credit card direct debit settles previous month's statement.
    settlement_day = _clamp_day(month_start.year, month_start.month, 7)
    settlement_date = _shift_to_weekday(
        date(month_start.year, month_start.month, settlement_day)
    )
    if settlement_date <= month_end and credit_card_settlement > 0:
        entries.append(
            Tx(settlement_date, "HALIFAX CREDIT CARD", "DD",
               money_out=round(credit_card_settlement, 2))
        )

    # Salary on the last working day of the month (BGC = Bank Giro Credit).
    payday = month_end
    while payday.weekday() >= 5:
        payday -= timedelta(days=1)
    entries.append(
        Tx(payday, f"{PERSONA['employer']} SALARY", "BGC",
           money_in=PERSONA["monthly_net_salary"])
    )

    # Variable debit-card spend from the LLM payload.
    for item in spend.debit:
        try:
            day = _clamp_day(month_start.year, month_start.month, int(item["day"]))
            d = date(month_start.year, month_start.month, day)
            if d > month_end:
                continue
            entries.append(Tx(
                d,
                str(item["description"])[:30].upper(),
                str(item.get("type", "DEB")).upper(),
                money_out=float(item["amount"]),
            ))
        except (KeyError, ValueError, TypeError):
            continue

    # Save-the-change daily rounding (Halifax product on the real statement).
    n_save = rng.randint(*SAVE_THE_CHANGE_PER_MONTH)
    for _ in range(n_save):
        d = date(month_start.year, month_start.month,
                 rng.randint(1, (month_end - month_start).days + 1))
        if d > month_end:
            continue
        entries.append(Tx(d, "SAVETHECHANGE-6366", "BP",
                          money_out=round(rng.uniform(0.05, 1.95), 2)))

    # Monthly interest (gross) on the 1st.
    interest = round(opening_balance * 0.001, 2)
    if interest > 0:
        entries.append(Tx(month_start, "INTEREST (GROSS)", "",
                          money_in=interest))

    # Sort by date (stable: within a date, money-out before money-in matches
    # the real samples where bills hit before salary credits).
    entries.sort(key=lambda t: (t.date, 0 if t.money_out else 1))

    # Walk the ledger to compute running balances.
    bal = opening_balance
    for tx in entries:
        bal += tx.money_in or 0.0
        bal -= tx.money_out or 0.0
        tx.balance = round(bal, 2)
    return entries


def _render_savings_md(
    month_start: date,
    month_end: date,
    opening_balance: float,
    closing_balance: float,
    money_in_total: float,
    money_out_total: float,
    entries: list[Tx],
) -> str:
    period = (f"{month_start.strftime('%d %B %Y')} to "
              f"{month_end.strftime('%d %B %Y')}")
    # YAML frontmatter — read by src.ingestion.parse_statements for account
    # metadata. The body that follows mirrors the look of a real statement.
    header = f"""---
account_id: "{PERSONA['savings_account']}"
account_type: "current"
institution: "Halifax"
period_start: "{month_start.isoformat()}"
period_end: "{month_end.isoformat()}"
---

# Halifax — REWARD CURRENT ACCOUNT

Document requested by: {PERSONA['name']} {PERSONA['address_line']} {PERSONA['town']} {PERSONA['county']} {PERSONA['postcode']}

Sort Code: {PERSONA['sort_code']} | Account Number: {PERSONA['savings_account']}

## {period}

Balance on {month_start.strftime('%d %B %Y')}: £{opening_balance:,.2f}
Money In: £{money_in_total:,.2f}
Money Out: £{money_out_total:,.2f}
Balance on {month_end.strftime('%d %B %Y')}: £{closing_balance:,.2f}

## Your Transactions

| Date | Description | Amount | Balance |
|------|-------------|--------|---------|
"""

    rows = []
    for tx in entries:
        # Signed amount: positive = money in, negative = money out.
        signed = (tx.money_in or 0.0) - (tx.money_out or 0.0)
        rows.append(
            f"| {tx.date.isoformat()} "
            f"| {tx.description} ({tx.tx_type}) "
            f"| {signed:.2f} "
            f"| {tx.balance:.2f} |"
        )

    footer = f"""

## Transaction types

BGC: Bank Giro Credit | BP: Bill Payment | CHG: Charge | CHQ: Cheque |
COR: Correction | CPT: Cashpoint | DD: Direct Debit | DEB: Debit Card |
DEP: Deposit | FEE: Fixed Service | FPI: Faster Payment In |
FPO: Faster Payment Out | MPI: Mobile Payment In | MPO: Mobile Payment Out |
PAY: Payment | SO: Standing Order | TFR: Transfer

If you think something is incorrect, please contact us on {PERSONA['support_phone']}.

Halifax is a division of Bank of Scotland plc. (Synthetic sample — not a real statement.)
"""
    return header + "\n".join(rows) + footer


# ---------------------------------------------------------------------------
# Credit card statement assembly.
# ---------------------------------------------------------------------------

def _build_credit_ledger(
    statement_period_start: date,
    statement_period_end: date,
    spend: MonthlySpend,
    rng: random.Random,
) -> list[CardTx]:
    """Build credit card transactions; statement period is mid-month → mid-month."""
    txns: list[CardTx] = []
    days_in_period = (statement_period_end - statement_period_start).days + 1

    for item in spend.credit:
        try:
            offset = rng.randint(0, max(0, days_in_period - 1))
            tx_date = statement_period_start + timedelta(days=offset)
            posted = tx_date + timedelta(days=rng.choice([0, 1, 1, 2]))
            if posted > statement_period_end:
                posted = statement_period_end
            txns.append(CardTx(
                tx_date=tx_date,
                posted_date=posted,
                description=str(item["description"])[:38].upper(),
                amount=float(item["amount"]),
            ))
        except (KeyError, ValueError, TypeError):
            continue

    txns.sort(key=lambda t: (t.posted_date, t.tx_date))
    return txns


def _render_credit_md(
    statement_date: date,
    period_start: date,
    period_end: date,
    previous_balance: float,
    payments_received: float,
    new_spend: float,
    new_balance: float,
    txns: list[CardTx],
) -> str:
    payment_due_date = _shift_to_weekday(statement_date + timedelta(days=26))
    minimum_payment = max(5.00, round(new_balance * 0.01, 2))
    available = PERSONA["credit_limit"] - new_balance
    est_interest = round(new_balance * 0.0247, 2)

    # Running balance through the period — starts at previous_balance, the
    # payment-received line zeroes the carry-over, then purchases accumulate.
    running = previous_balance
    tx_rows: list[str] = []

    # Opening carry-over (only emit if non-zero so we don't pollute the first
    # statement with a zero-balance line).
    if previous_balance > 0:
        tx_rows.append(
            f"| {period_start.isoformat()} "
            f"| BALANCE FROM PREVIOUS STATEMENT "
            f"| 0.00 "
            f"| {running:.2f} |"
        )
        running -= payments_received
        tx_rows.append(
            f"| {period_start.isoformat()} "
            f"| DIRECT DEBIT PAYMENT - THANK YOU "
            f"| {payments_received:.2f} "
            f"| {running:.2f} |"
        )

    for tx in txns:
        signed = tx.amount if tx.is_credit else -tx.amount
        running += signed
        tx_rows.append(
            f"| {tx.posted_date.isoformat()} "
            f"| {tx.description} "
            f"| {signed:.2f} "
            f"| {running:.2f} |"
        )

    header = f"""---
account_id: "{PERSONA['credit_card_last4']}"
account_type: "credit_card"
institution: "Halifax Clarity"
period_start: "{period_start.isoformat()}"
period_end: "{period_end.isoformat()}"
---

# Halifax Clarity Credit Card Statement

{PERSONA['name']} {PERSONA['address_line']} {PERSONA['town']} {PERSONA['county']} {PERSONA['postcode']}

## Your credit card statement {statement_date.strftime('%d %B %Y')}

## Transactions

| Date | Description | Amount | Balance |
|------|-------------|--------|---------|
"""

    rows = tx_rows

    footer = f"""

## Account summary

| Mastercard number | {PERSONA['card_number']} |
|---|---|
| Cardholder | {PERSONA['name']} |
| Your credit limit | £{PERSONA['credit_limit']:,.2f} |
| Available to spend | £{available:,.2f} |
| Next month's estimated interest | £{est_interest:,.2f} |
| Previous balance | £{previous_balance:,.2f} |
| Payments received | £{payments_received:,.2f} |
| New transactions, fees and charges | £{new_spend:,.2f} |
| Your new balance | £{new_balance:,.2f} |
| Minimum payment due | £{minimum_payment:,.2f} |
| To reach your account by | {payment_due_date.strftime('%d %B %Y')} |

Current standard interest rates: 24.76% p.a. (variable). We'll take your
Direct Debit of £{new_balance:,.2f} from your bank account on
{payment_due_date.strftime('%d/%m/%y')}.

## Breakdown of balance

| Balance Type | Effective Annual Rate (%) | Outstanding Balance (£) | Interest Charged (£) |
|--------------|---------------------------|-------------------------|----------------------|
| Purchases (Standard) | 24.76 | {new_balance:,.2f} | 0.00 |

CASHBACK earned this month: £0.00 | Cashback balance: £0.00

Halifax is a division of Bank of Scotland plc. Customer Services: {PERSONA['card_services_phone']}. (Synthetic sample — not a real statement.)
"""
    return header + "\n".join(rows) + footer


# ---------------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------------

def generate_all(
    start: date,
    end: date,
    out_dir: Path,
    seed: int = 1588,
) -> dict[str, list[Path]]:
    """Generate every savings + credit-card statement between ``start`` and ``end``.

    Returns a dict like ``{"savings": [...], "credit": [...]}`` listing every
    file written.
    """
    rng = random.Random(seed)
    savings_dir = out_dir / "savings_stmt"
    credit_dir = out_dir / "crdit_stmt"
    savings_dir.mkdir(parents=True, exist_ok=True)
    credit_dir.mkdir(parents=True, exist_ok=True)

    written = {"savings": [], "credit": []}

    opening_balance = OPENING_SAVINGS_BALANCE
    previous_card_balance = 0.0  # First statement starts clean.

    for month_start, month_end in iter_months(start, end):
        month_label = month_start.strftime("%B %Y")
        spend = _llm_monthly_spend(month_label, seed=seed + month_start.toordinal())

        # --- savings statement ----------------------------------------------
        entries = _build_savings_ledger(
            month_start=month_start,
            month_end=month_end,
            opening_balance=opening_balance,
            spend=spend,
            credit_card_settlement=previous_card_balance,
            rng=rng,
        )
        money_in_total = round(sum(t.money_in or 0 for t in entries), 2)
        money_out_total = round(sum(t.money_out or 0 for t in entries), 2)
        closing = entries[-1].balance if entries else opening_balance

        savings_md = _render_savings_md(
            month_start, month_end,
            opening_balance, closing,
            money_in_total, money_out_total,
            entries,
        )
        savings_path = savings_dir / f"{month_start.year}_{month_start.strftime('%B')}_Statement.md"
        savings_path.write_text(savings_md, encoding="utf-8")
        written["savings"].append(savings_path)

        # --- credit card statement ------------------------------------------
        statement_date = date(
            month_start.year, month_start.month,
            _clamp_day(month_start.year, month_start.month, 12),
        )
        period_start = statement_date - timedelta(days=30)
        period_end = statement_date

        card_txns = _build_credit_ledger(period_start, period_end, spend, rng)
        new_spend = round(sum(t.amount for t in card_txns if not t.is_credit), 2)
        payments_received = round(previous_card_balance, 2)
        new_balance = round(previous_card_balance - payments_received + new_spend, 2)

        credit_md = _render_credit_md(
            statement_date=statement_date,
            period_start=period_start,
            period_end=period_end,
            previous_balance=previous_card_balance,
            payments_received=payments_received,
            new_spend=new_spend,
            new_balance=new_balance,
            txns=card_txns,
        )
        credit_path = credit_dir / (
            f"Statement_{PERSONA['credit_card_last4']}_"
            f"{month_start.strftime('%b-%y')}.md"
        )
        credit_path.write_text(credit_md, encoding="utf-8")
        written["credit"].append(credit_path)

        # roll state forward
        opening_balance = closing
        previous_card_balance = new_balance

    return written
