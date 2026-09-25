"""Common Alerting Protocol (CAP v1.2) XML Engine

Generates OASIS-compliant CAP 1.2 XML alerts for extreme weather anomalies.
Adheres strictly to the OASIS Standard CAP-v1.2 specification:
- 4-Tier Severity Classification: Minor, Moderate, Severe, Extreme
- Event details, certainty, urgency, instruction, and affected polygon area
- Multilingual content support (English, Hindi, and Odia regional templates)
- Optional Groq LLM integration for dynamic meteorological bulletins
"""

import os
import json
import uuid
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from xml.sax.saxutils import escape
from typing import Dict, Any, List, Optional
from config import ALERT_CONFIG

logger = logging.getLogger("chakranet.cap_engine")


class CAPAlertEngine:
    """Formats and validates CAP 1.2 XML alerts with multilingual extensions."""

    def __init__(self):
        self.sender_id = "in.gov.moes.ncmrwf.chakranet"
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = "qwen/qwen3.8-27b"

    def classify_severity(self, max_rain_mm: float, prob_exceed: float) -> str:
        """Determines 4-level CAP severity based on rainfall amount and probability."""
        if prob_exceed >= ALERT_CONFIG.prob_extreme or max_rain_mm >= 200.0:
            return "Extreme"
        elif prob_exceed >= ALERT_CONFIG.prob_severe or max_rain_mm >= 115.0:
            return "Severe"
        elif prob_exceed >= ALERT_CONFIG.prob_moderate or max_rain_mm >= 65.0:
            return "Moderate"
        else:
            return "Minor"

    def generate_cap_xml(
        self,
        event_name: str,
        lead_time_hours: int,
        severity: str,
        affected_area_desc: str,
        coordinates: List[List[float]],
        language: str = "en-US",
        headline: Optional[str] = None,
        description: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """Constructs an OASIS CAP 1.2 XML document string."""
        identifier = f"CHAKRANET-{uuid.uuid4().hex[:12].upper()}"
        sent_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        
        # Color & urgency mappings
        urgency = "Immediate" if severity in ["Extreme", "Severe"] else "Expected"
        certainty = "Observed" if lead_time_hours <= 24 else "Likely"
        
        if not headline:
            headline = f"ChakraNet Warning: {severity} Cyclone Hazard ({event_name}) at T+{lead_time_hours}h"
        
        if not description:
            description = (
                f"Severe extreme-weather anomaly tracked by ChakraNet atmospheric forecasting pipeline. "
                f"Anticipated high-impact heavy precipitation and gale-force wind gust risk "
                f"over {affected_area_desc} at lead time +{lead_time_hours} hours. "
                f"Classified at severity level '{severity}'."
            )
            
        if not instruction:
            instruction = (
                "Take immediate precautionary shelter. Avoid low-lying coastal terrain and sea voyages. "
                "Follow official advisories broadcasted by IMD, NDMA, and State Disaster Management Authorities."
            )

        # Format polygon coordinate string: "lat1,lon1 lat2,lon2 ..."
        polygon_str = " ".join([f"{lat:.4f},{lon:.4f}" for lat, lon in coordinates])

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>{escape(identifier)}</identifier>
  <sender>{escape(self.sender_id)}</sender>
  <sent>{sent_time}</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>{escape(language)}</language>
    <category>Met</category>
    <event>{escape(event_name)}</event>
    <urgency>{escape(urgency)}</urgency>
    <severity>{escape(severity)}</severity>
    <certainty>{escape(certainty)}</certainty>
    <eventCode>
      <valueName>ChakraNetAlertCode</valueName>
      <value>CHAKRA-MET-{severity.upper()}</value>
    </eventCode>
    <headline>{escape(headline)}</headline>
    <description>{escape(description)}</description>
    <instruction>{escape(instruction)}</instruction>
    <contact>NCMRWF Emergency Operational Cell (support@ncmrwf.gov.in)</contact>
    <area>
      <areaDesc>{escape(affected_area_desc)}</areaDesc>
      <polygon>{polygon_str}</polygon>
    </area>
  </info>
</alert>"""
        return xml.strip()

    def generate_bhashini_hindi_translation(
        self,
        event_name: str,
        severity: str,
        lead_time_hours: int,
        affected_area_desc: str,
    ) -> Dict[str, str]:
        """Provides simulated Bhashini multilingual templated translation in Hindi."""
        severity_hi = {
            "Extreme": "अत्यधिक गंभीर (Extreme)",
            "Severe": "गंभीर (Severe)",
            "Moderate": "मध्यम (Moderate)",
            "Minor": "सामान्य (Minor)",
        }.get(severity, severity)

        headline_hi = f"चक्रवात चेतावनी: {severity_hi} मौसम खतरा ({event_name}) T+{lead_time_hours} घंटे"
        description_hi = (
            f"चक्रवातनेट (ChakraNet) वायुमंडलीय विश्लेषण प्रणाली द्वारा ट्रैक किया गया मौसम खतरा। "
            f"{affected_area_desc} क्षेत्र में आगामी +{lead_time_hours} घंटों में मूसलाधार वर्षा "
            f"और तेज हवाओं की चेतावनी। गंभीरता स्तर: '{severity_hi}'।"
        )
        instruction_hi = (
            "तुरंत सुरक्षित स्थानों पर शरण लें। निचले तटीय क्षेत्रों और समुद्र में जाने से बचें। "
            "आईएमडी (IMD) और राज्य आपदा प्रबंधन प्राधिकरण (SDMA) के आधिकारिक निर्देशों का पालन करें।"
        )

        return {
            "language": "hi-IN",
            "provider": "SIMULATED Bhashini",
            "headline": headline_hi,
            "description": description_hi,
            "instruction": instruction_hi,
        }

    def generate_odia_translation(
        self,
        event_name: str,
        severity: str,
        lead_time_hours: int,
        affected_area_desc: str,
    ) -> Dict[str, str]:
        """Provides Odia regional translation for Odisha coastal communities."""
        severity_or = {
            "Extreme": "ଅତ୍ୟନ୍ତ ଗମ୍ଭୀର (Extreme)",
            "Severe": "ଗମ୍ଭୀର (Severe)",
            "Moderate": "ମଧ୍ୟମ (Moderate)",
            "Minor": "ସାଧାରଣ (Minor)",
        }.get(severity, severity)

        headline_or = f"ବାତ୍ୟା ସତର୍କତା: {severity_or} ବିପଦ ({event_name}) T+{lead_time_hours} ଘଣ୍ଟା"
        description_or = (
            f"ଚକ୍ରବାତନେଟ୍ ବାୟୁମଣ୍ଡଳୀୟ ପୂର୍ବାନୁମାନ ବ୍ୟବସ୍ଥା ଦ୍ୱାରା {affected_area_desc} ଅଞ୍ଚଳରେ "
            f"ଆଗାମୀ +{lead_time_hours} ଘଣ୍ଟା ମଧ୍ୟରେ ପ୍ରବଳ ବର୍ଷା ଓ ଝଡ଼ ପବନର ସତର୍କ ସୂଚନା। "
            f"ବିପଦ ସ୍ତର: '{severity_or}'।"
        )
        instruction_or = (
            "ତୁରନ୍ତ ନିକଟସ୍ଥ ବାତ୍ୟା ଆଶ୍ରୟସ୍ଥଳକୁ ଯାଆନ୍ତୁ। ସମୁଦ୍ରକୂଳ ଓ ତଳିଆ ଅଞ୍ଚଳରୁ ଦୂରେଇ ରୁହନ୍ତୁ। "
            "ଓଡ଼ିଶା ରାଜ୍ୟ ବିପର୍ଯ୍ୟୟ ପରିଚାଳନା ପ୍ରାଧିକରଣ (OSDMA) ର ନିର୍ଦ୍ଦେଶାବଳୀ ପାଳନ କରନ୍ତୁ।"
        )

        return {
            "language": "or-IN",
            "provider": "Bhashini Regional (Odia)",
            "headline": headline_or,
            "description": description_or,
            "instruction": instruction_or,
        }

    def generate_groq_bulletin(
        self,
        event_name: str,
        severity: str,
        lead_time_hours: int,
        affected_area_desc: str,
        language: str = "en",
    ) -> Optional[Dict[str, str]]:
        """Invokes Groq API (qwen/qwen3.8-27b) for real-time dynamic advisory generation."""
        if not self.groq_api_key:
            return None

        prompt = (
            f"You are an official meteorologist at NCMRWF/MoES India. Generate a concise, clear {severity} "
            f"cyclone bulletin for {event_name} affecting {affected_area_desc} at forecast lead time T+{lead_time_hours} hours. "
            f"Language: {language}. Return strictly JSON with keys: headline, description, instruction. "
            f"Do not include any other markdown formatting or text outside the JSON."
        )

        try:
            req_data = json.dumps({
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": "You are a professional weather authority. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=req_data,
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (ChakraNet/1.0)",
                }
            )

            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"].strip()
                # Parse JSON if enclosed in markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                parsed = json.loads(content)
                parsed["provider"] = f"Groq ({self.groq_model})"
                return parsed
        except Exception as e:
            logger.warning(f"Groq API call failed or timed out: {e}. Falling back to deterministic template.")
            return None
