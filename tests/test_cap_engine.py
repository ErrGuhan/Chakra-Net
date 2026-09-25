"""Unit tests for ChakraNet OASIS CAP 1.2 XML Engine"""

import pytest
import xml.etree.ElementTree as ET
from serving.api.cap_engine import CAPAlertEngine


def test_classify_severity():
    engine = CAPAlertEngine()

    # Extreme: prob >= 0.85 or rain >= 200 mm
    assert engine.classify_severity(max_rain_mm=210.0, prob_exceed=0.60) == "Extreme"
    assert engine.classify_severity(max_rain_mm=150.0, prob_exceed=0.90) == "Extreme"

    # Severe: prob >= 0.65 or rain >= 115 mm
    assert engine.classify_severity(max_rain_mm=130.0, prob_exceed=0.50) == "Severe"
    assert engine.classify_severity(max_rain_mm=80.0, prob_exceed=0.70) == "Severe"

    # Moderate: prob >= 0.40 or rain >= 65 mm
    assert engine.classify_severity(max_rain_mm=75.0, prob_exceed=0.30) == "Moderate"
    assert engine.classify_severity(max_rain_mm=40.0, prob_exceed=0.45) == "Moderate"

    # Minor
    assert engine.classify_severity(max_rain_mm=30.0, prob_exceed=0.20) == "Minor"


def test_generate_cap_xml_structure():
    engine = CAPAlertEngine()
    coords = [[84.8, 19.1], [85.2, 19.1], [85.2, 19.6], [84.8, 19.6], [84.8, 19.1]]

    xml_str = engine.generate_cap_xml(
        event_name="Cyclonic Storm Phailin",
        lead_time_hours=48,
        severity="Severe",
        affected_area_desc="Ganjam Coastal Belt",
        coordinates=coords,
        language="en-US",
        headline="High Wind and Extreme Rain Warning",
        description="Destructive convective eyewall winds approaching coast.",
        instruction="Evacuate coastal vulnerable dwellings immediately.",
    )

    # Must be valid XML
    root = ET.fromstring(xml_str)
    assert "alert" in root.tag

    # Strip namespace if present
    def find_tag(parent, tag_name):
        for elem in parent.iter():
            if elem.tag.endswith(tag_name):
                return elem
        return None

    assert find_tag(root, "identifier") is not None
    assert find_tag(root, "sender") is not None
    assert find_tag(root, "status").text == "Actual"
    assert find_tag(root, "msgType").text == "Alert"
    assert find_tag(root, "scope").text == "Public"

    info = find_tag(root, "info")
    assert info is not None
    assert find_tag(info, "category").text == "Met"
    assert find_tag(info, "event").text == "Cyclonic Storm Phailin"
    assert find_tag(info, "urgency").text == "Immediate"
    assert find_tag(info, "severity").text == "Severe"
    # At lead time 48h, certainty is "Likely"
    assert find_tag(info, "certainty").text == "Likely"

    # Test short lead time <= 24h gives "Observed"
    xml_short = engine.generate_cap_xml(
        event_name="Cyclonic Storm Phailin",
        lead_time_hours=12,
        severity="Extreme",
        affected_area_desc="Ganjam Coastal Belt",
        coordinates=coords,
    )
    root_short = ET.fromstring(xml_short)
    assert find_tag(find_tag(root_short, "info"), "certainty").text == "Observed"

    area = find_tag(info, "area")
    assert area is not None
    assert find_tag(area, "areaDesc").text == "Ganjam Coastal Belt"
    poly = find_tag(area, "polygon")
    assert poly is not None
    assert len(poly.text.strip().split()) >= 4


def test_multilingual_translations():
    engine = CAPAlertEngine()

    tmpl_hi = engine.generate_bhashini_hindi_translation(
        event_name="Phailin",
        severity="Extreme",
        lead_time_hours=48,
        affected_area_desc="Ganjam District",
    )
    assert tmpl_hi["language"] == "hi-IN"
    assert "headline" in tmpl_hi
    assert "instruction" in tmpl_hi
    assert len(tmpl_hi["headline"]) > 0

    tmpl_or = engine.generate_odia_translation(
        event_name="Phailin",
        severity="Extreme",
        lead_time_hours=48,
        affected_area_desc="Ganjam District",
    )
    assert tmpl_or["language"] == "or-IN"
    assert "headline" in tmpl_or
    assert "description" in tmpl_or
    assert len(tmpl_or["headline"]) > 0


def test_xml_character_escaping():
    engine = CAPAlertEngine()
    coords = [[84.0, 19.0], [85.0, 19.0], [85.0, 20.0], [84.0, 20.0], [84.0, 19.0]]

    # Headline and description containing XML special characters
    special_text = "Severe Wind & Rain <Warning> \"Cat-4\" & 100% Risk"
    xml_str = engine.generate_cap_xml(
        event_name="Test & Storm",
        lead_time_hours=24,
        severity="Extreme",
        affected_area_desc="Coast & Estuary",
        coordinates=coords,
        headline=special_text,
        description=special_text,
        instruction=special_text,
    )

    # Must parse cleanly without XML syntax errors
    root = ET.fromstring(xml_str)
    assert root is not None
