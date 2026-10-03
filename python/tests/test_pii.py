"""Tests for sentrygate's deterministic PII detection, masking and redaction.

stdlib + pytest only. These exercise the on-device path, which must work with
no model and without importing the scanner engine.
"""
import sys

import sentrygate
from sentrygate import pii


def test_valid_visa_masks_as_credit_card():
    text = "my card is 4111111111111111 thanks"
    spans = pii.detect(text)
    types = {s.type for s in spans}
    assert "CREDIT_CARD" in types
    r = pii.mask(text)
    assert "<CREDIT_CARD_1>" in r.masked_text
    assert "4111111111111111" not in r.masked_text


def test_invalid_card_fails_luhn_not_masked():
    text = "my card is 4111111111111112 thanks"
    spans = pii.detect(text)
    assert "CREDIT_CARD" not in {s.type for s in spans}
    r = pii.mask(text)
    assert "4111111111111112" in r.masked_text


def test_email_phone_ssn_ipv4_detection():
    text = ("reach me at jane.doe@example.com or 415-555-0199, "
            "ssn 123-45-6789, server 192.168.1.1")
    types = {s.type for s in pii.detect(text)}
    assert "EMAIL" in types
    assert "PHONE" in types
    assert "SSN" in types
    assert "IPV4" in types


def test_mask_then_unmask_roundtrips_exactly():
    text = ("email jane.doe@example.com call 415-555-0199 card "
            "4111111111111111 ssn 123-45-6789 ip 10.0.0.1")
    r = pii.mask(text)
    assert r.masked_text != text
    restored = pii.unmask(r.masked_text, r.mapping)
    assert restored == text


def test_same_value_gets_same_placeholder():
    text = "ping a@b.com then again a@b.com"
    r = pii.mask(text)
    # one unique placeholder, used twice
    assert r.masked_text.count("<EMAIL_1>") == 2
    assert "<EMAIL_2>" not in r.masked_text
    assert r.mapping["<EMAIL_1>"] == "a@b.com"


def test_redact_and_pii_guard_work_without_scanner_or_model():
    # Ensure no scanner/model is imported by the PII-only path.
    for mod in [m for m in list(sys.modules) if m.startswith(("sentrygate.scanner",
                "prompttest", "torch", "onnxruntime"))]:
        del sys.modules[mod]

    res = sentrygate.redact("contact a@b.com")
    assert "<EMAIL_1>" in res.masked_text

    g = sentrygate.Guard([sentrygate.PIIMasker()])
    gr = g.process("contact a@b.com or call 415-555-0199")
    assert gr.allowed is True
    assert "<EMAIL_1>" in gr.text
    assert g.unmask(gr.text) == "contact a@b.com or call 415-555-0199"

    # None of these should have pulled in the scanner or a model backend.
    assert "sentrygate.scanner" not in sys.modules
    assert "torch" not in sys.modules
    assert "onnxruntime" not in sys.modules
