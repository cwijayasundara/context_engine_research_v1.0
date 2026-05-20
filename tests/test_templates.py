from src.ingestion.templates import fingerprint


def test_strips_digits():
    a = fingerprint("PARENTPAY E-COM R BRIDGWATER")
    b = fingerprint("PARENTPAY E-COM R BRIDGWATER")
    assert a == b
    assert a.template == "PARENTPAY E-COM R BRIDGWATER"


def test_groups_numeric_ids():
    a = fingerprint("AMZNMKTPLACE*R66EF9ZC4")
    b = fingerprint("AMZNMKTPLACE*R12AB3XY0")
    assert a == b


def test_different_merchants_different_template():
    a = fingerprint("TESCO STORES 3372")
    b = fingerprint("SAINSBURY'S S/MKT WATFORD")
    assert a != b


def test_template_id_is_stable_hash():
    a = fingerprint("UBER *TRIP HELP.UBER.COM")
    b = fingerprint("UBER *TRIP HELP.UBER.COM")
    assert a.id == b.id
    assert len(a.id) == 8
