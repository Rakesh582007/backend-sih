"""
GiriKavach — CAP 1.2 Alert Payload Generator
==============================================

CAP (Common Alerting Protocol) 1.2 is the OASIS international standard
for emergency alerts. India's NDMA runs SACHET on it. Generating valid
CAP means your siren isn't a closed box — the same alert can be ingested
by national systems, TV/radio crawls, and cell broadcast without any
custom integration work.

------------------------------------------------------------------
SAFETY DECISION — READ THIS BEFORE CHANGING THE DEFAULT
------------------------------------------------------------------
`status` defaults to "Exercise", NOT "Actual".

CAP is a real standard consumed by real emergency infrastructure. A
payload marked status="Actual" is, by definition, claiming a genuine
emergency. During development, demos, and hackathon judging, that
claim would be false. Marking it "Exercise" is the honest and correct
setting, and it is also what CAP's own spec intends for drills.

Setting status="Actual" is therefore an explicit, deliberate argument
the caller must pass — never the default, and never something that
happens by accident during a demo.

------------------------------------------------------------------
STRUCTURE (CAP 1.2)
------------------------------------------------------------------
    <alert>          one message
      identifier, sender, sent, status, msgType, scope
      <info>         one per language/hazard
        category, event, urgency, severity, certainty
        onset, expires, senderName, headline, description, instruction
        <area>
          areaDesc, circle/polygon, geocode

Controlled vocabularies (CAP 1.2 spec — these are NOT free text):
  status   : Actual | Exercise | System | Test | Draft
  msgType  : Alert | Update | Cancel | Ack | Error
  scope    : Public | Restricted | Private
  category : Geo | Met | Safety | Security | Rescue | Fire | Health |
             Env | Transport | Infra | CBRNE | Other
  urgency  : Immediate | Expected | Future | Past | Unknown
  severity : Extreme | Severe | Moderate | Minor | Unknown
  certainty: Observed | Likely | Possible | Unlikely | Unknown
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"

# India Standard Time — CAP requires an explicit UTC offset, never naive time.
IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class AlertArea:
    """Where the alert applies. Circle is simplest for a village node."""
    description: str
    latitude: float
    longitude: float
    radius_km: float = 5.0
    # NDMA/SACHET keys off Indian census/LGD codes where available.
    geocode_name: str | None = None
    geocode_value: str | None = None


def _cap_timestamp(dt: datetime) -> str:
    """CAP 1.2 requires ISO 8601 WITH timezone offset (no 'Z' shorthand)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    stamp = dt.isoformat(timespec="seconds")
    # Python renders +05:30 correctly; CAP wants exactly that form.
    return stamp


def build_cap_alert(
    *,
    hazard: str,                      # "flood" or "landslide"
    risk_level: str,                  # NORMAL/WATCH/ALERT/CRITICAL
    probability: float,               # 0-100
    area: AlertArea,
    sender: str = "girikavach@example.org",
    sender_name: str = "GiriKavach Community Early Warning Node",
    status: str = "Exercise",         # see safety note in module docstring
    msg_type: str = "Alert",
    scope: str = "Public",
    expires_minutes: int = 60,
    sent: datetime | None = None,
    identifier: str | None = None,
    reasons: list[str] | None = None,
) -> str:
    """Return a CAP 1.2 XML document as a string.

    Raises ValueError on invalid controlled-vocabulary values — better to
    fail loudly here than emit a malformed alert a real system rejects.
    """
    valid_status = {"Actual", "Exercise", "System", "Test", "Draft"}
    valid_msgtype = {"Alert", "Update", "Cancel", "Ack", "Error"}
    valid_scope = {"Public", "Restricted", "Private"}
    if status not in valid_status:
        raise ValueError(f"status must be one of {sorted(valid_status)}, got {status!r}")
    if msg_type not in valid_msgtype:
        raise ValueError(f"msgType must be one of {sorted(valid_msgtype)}, got {msg_type!r}")
    if scope not in valid_scope:
        raise ValueError(f"scope must be one of {sorted(valid_scope)}, got {scope!r}")

    hazard = hazard.lower()
    if hazard not in {"flood", "landslide"}:
        raise ValueError(f"hazard must be 'flood' or 'landslide', got {hazard!r}")

    sent = sent or datetime.now(IST)
    identifier = identifier or f"GIRIKAVACH-{uuid.uuid4().hex[:12].upper()}"

    # --- map internal risk tier onto CAP's controlled vocabulary ---
    # Landslides are geophysical; flash floods are meteorological.
    category = "Geo" if hazard == "landslide" else "Met"
    event = "Landslide" if hazard == "landslide" else "Flash Flood"

    severity_map = {
        "CRITICAL": ("Extreme", "Immediate", "Likely"),
        "ALERT": ("Severe", "Immediate", "Likely"),
        "WATCH": ("Moderate", "Expected", "Possible"),
        "NORMAL": ("Minor", "Future", "Unlikely"),
    }
    severity, urgency, certainty = severity_map.get(
        risk_level.upper(), ("Unknown", "Unknown", "Unknown")
    )

    # Certainty is raised only when the model is very confident AND the
    # tier is already severe — we do not claim "Observed" from a
    # prediction. "Observed" would mean a human/instrument confirmed an
    # actual event, which a probability alone never does.
    if probability >= 95 and risk_level.upper() == "CRITICAL":
        certainty = "Likely"

    ET.register_namespace("", CAP_NS)
    alert = ET.Element(f"{{{CAP_NS}}}alert")

    def sub(parent, tag, text):
        el = ET.SubElement(parent, f"{{{CAP_NS}}}{tag}")
        el.text = str(text)
        return el

    sub(alert, "identifier", identifier)
    sub(alert, "sender", sender)
    sub(alert, "sent", _cap_timestamp(sent))
    sub(alert, "status", status)
    sub(alert, "msgType", msg_type)
    sub(alert, "scope", scope)

    info = ET.SubElement(alert, f"{{{CAP_NS}}}info")
    sub(info, "language", "en-IN")
    sub(info, "category", category)
    sub(info, "event", event)
    sub(info, "urgency", urgency)
    sub(info, "severity", severity)
    sub(info, "certainty", certainty)
    sub(info, "onset", _cap_timestamp(sent))
    sub(info, "expires", _cap_timestamp(sent + timedelta(minutes=expires_minutes)))
    sub(info, "senderName", sender_name)

    headline = f"{risk_level.upper()}: {event} risk {probability:.0f}% near {area.description}"
    sub(info, "headline", headline)

    description = (
        f"Local sensor network estimates {probability:.0f}% probability of "
        f"{event.lower()} conditions within the monitored catchment. "
        f"Assessment computed on-site from rainfall, soil saturation, "
        f"slope inclination and river level telemetry."
    )
    if reasons:
        description += " Contributing factors: " + "; ".join(reasons) + "."
    if status != "Actual":
        description = f"[{status.upper()} — NOT A REAL EMERGENCY] " + description
    sub(info, "description", description)

    if hazard == "landslide":
        instruction = (
            "Move away from steep slopes, gullies and stream channels immediately. "
            "Do not shelter in buildings at the base of a slope. Move to higher "
            "stable ground away from the flow path. Follow instructions from local "
            "authorities and NDRF personnel."
        )
    else:
        instruction = (
            "Move immediately to higher ground away from the river channel. "
            "Do not attempt to cross flowing water on foot or by vehicle. "
            "Follow instructions from local authorities and NDRF personnel."
        )
    if status != "Actual":
        instruction = f"[{status.upper()} — NO ACTION REQUIRED] " + instruction
    sub(info, "instruction", instruction)

    area_el = ET.SubElement(info, f"{{{CAP_NS}}}area")
    sub(area_el, "areaDesc", area.description)
    # CAP circle format: "lat,lon radius_km"
    sub(area_el, "circle", f"{area.latitude},{area.longitude} {area.radius_km}")
    if area.geocode_name and area.geocode_value:
        geo = ET.SubElement(area_el, f"{{{CAP_NS}}}geocode")
        sub(geo, "valueName", area.geocode_name)
        sub(geo, "value", area.geocode_value)

    ET.indent(alert, space="  ")
    xml_body = ET.tostring(alert, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body


if __name__ == "__main__":
    demo_area = AlertArea(
        description="Meppadi Panchayat, Wayanad District, Kerala",
        latitude=11.4654,
        longitude=76.1358,
        radius_km=5.0,
        geocode_name="LGD_CODE",
        geocode_value="000000",
    )
    print(
        build_cap_alert(
            hazard="landslide",
            risk_level="CRITICAL",
            probability=91.4,
            area=demo_area,
            reasons=["deep soil saturation 93%", "slope 41 deg", "sustained tilt creep"],
        )
    )
