from src.ingestion.locations import parse_location, Location


def test_uk_town_no_country():
    loc = parse_location("SAINSBURY'S S/MKT WATFORD")
    assert loc == Location(id="watford", name="Watford", country="GB")


def test_us_state_code():
    loc = parse_location("GITHUB, INC. SAN FRANCISCO CA")
    assert loc == Location(id="san-francisco-ca", name="San Francisco CA", country="US")


def test_explicit_country_code():
    loc = parse_location("LUXE WATCH STORE PARIS FR")
    assert loc == Location(id="paris-fr", name="Paris FR", country="FR")


def test_apple_billing_country():
    loc = parse_location("APPLE.COM/BILL CORK IRL")
    assert loc == Location(id="cork-irl", name="Cork IRL", country="IE")


def test_unknown_returns_none():
    assert parse_location("BALANCE FROM PREVIOUS STATEMENT") is None
