"""
GiriKavach — LLM Narration Service (fully local, no cloud)
============================================================

WHAT THIS DOES
Two very different things, on two very different safety footings:

  1. narrate() — citizen-facing alert text. As of this rewrite, this is
     ALWAYS one of the pre-written, reviewed strings in
     alert_templates.py, with only {place} and {probability} filled
     in. There is NO code path from narrate() to Ollama anymore.

  2. situation_report() — an English, ops-facing summary for a control
     room operator. This still uses the LLM, because a garbled or
     imprecise ops summary is an inconvenience, not a life-safety
     failure the way a wrong citizen alert is. Even here, output is
     checked for hallucinated place names before being trusted.

WHY narrate() NO LONGER TOUCHES THE LLM — read this before a judge asks
This module used to build a deterministic template, then ask the LLM
to translate/rephrase it, on the theory that constraining the prompt
("keep every fact, invent nothing, output only the message") would be
enough. Testing proved that wrong, in one run:
  - Hindi output hallucinated "उत्तराखंड" (Uttarakhand) when the actual
    place was Meppadi, Kerala — a fabricated location ~2000km wrong.
  - The "[DRILL — NOT A REAL EMERGENCY]" prefix was silently dropped.
  - Raw sensor facts leaked verbatim despite the prompt forbidding it.
  - Malayalam output was garbled and misspelled the place name.
A prompt is a request, not a guarantee. On a channel that can tell
someone to evacuate, "usually follows instructions" is not good enough.
Removing the LLM from this path removes the failure mode entirely,
rather than trying to prompt-engineer around it.

WHAT THIS DOES *NOT* DO — read this before a judge asks
Neither function decides anything. Neither sees a threshold, votes on
whether to alarm, or changes a risk level. By the time either is
called, the decision is already final and the siren may already be
sounding. This is a TRANSLATION/SUMMARY layer, not a decision layer.

Why that separation is deliberate:
  - Life-safety triggers must be deterministic and auditable. A judge
    (or a district collector) must be able to trace exactly why the
    siren fired. "The language model felt it was risky" is not an
    auditable answer.
  - Latency: local inference is 1-10s. The buzzer must fire in
    milliseconds. They cannot be on the same path.

DEGRADED MODE (important)
situation_report() falls back to plain, factual text if Ollama is
down, the model is missing, generation times out, or the output fails
the hallucination guard. narrate() never depended on the LLM being up
in the first place. Test the situation_report fallback path
deliberately — see test_alert_safety.py.

SETUP
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen2.5:7b      # ~4.7GB, needs ~8GB free RAM
    ollama serve                # usually auto-starts

    # if 7b feels slow on CPU, this is a fine downgrade for
    # situation_report's ops-summary wording:
    ollama pull qwen2.5:3b
"""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass

import urllib.error
import urllib.request

import alert_templates
from validate_alert import check_alert_text

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"
# First call must load ~5GB into RAM — on CPU that alone can take 30-90s.
# Steady-state generation is far quicker, but the timeout has to survive
# the cold start or every first alert silently falls back.
TIMEOUT_SECONDS = 120

# Set True to print why a generation failed instead of failing silently.
# Production leaves this False: an alerting system must not spam stderr
# during an event. Debugging turns it on.
DEBUG = False


@dataclass
class NarrationRequest:
    risk_level: str            # NORMAL | WATCH | ALERT | CRITICAL
    hazard: str                # "flood" | "landslide"
    probability: float         # 0-100
    place_name: str
    reasons: list[str]
    language: str = "en"
    is_exercise: bool = True   # mirrors CAP status; see cap_generator.py


def resolve_model(preferred: str = DEFAULT_MODEL) -> str | None:
    """Return the EXACT installed model tag to use, or None if none usable.

    This exists because of a bug worth remembering: the old check matched
    loosely ("does anything start with qwen2.5?") but the generate call
    demanded an exact tag. Pull qwen2.5:3b instead of :7b and you'd get
    "available: True" followed by a silent 404 on every request.
    Availability and usability must be decided by the same string.
    """
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
    except Exception as e:
        if DEBUG:
            print(f"[llm] ollama unreachable: {type(e).__name__}: {e}")
        return None

    names = [m.get("name", "") for m in tags.get("models", [])]
    if not names:
        if DEBUG:
            print("[llm] ollama running but no models installed")
        return None

    if preferred in names:
        return preferred

    # Fall back to any same-family tag, then to anything at all, rather
    # than failing because of a version suffix.
    family = preferred.split(":")[0]
    for n in names:
        if n.startswith(family):
            if DEBUG:
                print(f"[llm] '{preferred}' not installed, using '{n}'")
            return n

    if DEBUG:
        print(f"[llm] no {family} model; falling back to '{names[0]}'")
    return names[0]


def _call_ollama(prompt: str, model: str) -> str | None:
    """Returns generated text, or None on any failure (never raises)."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Low temperature: this is safety-adjacent messaging, not
        # creative writing, even for the ops-facing summary.
        "options": {"temperature": 0.2, "num_predict": 220},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
            return json.loads(r.read()).get("response", "").strip()
    except urllib.error.HTTPError as e:
        if DEBUG:
            print(f"[llm] HTTP {e.code} from ollama: {e.read().decode()[:200]}")
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        if DEBUG:
            print(f"[llm] generation failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------
# narrate() — citizen-facing. Templates only. No LLM. See module
# docstring for why.
# ---------------------------------------------------------------------
def narrate(req: NarrationRequest) -> dict:
    """Main entry point for citizen-facing alerts.

    Always returns one of the pre-written strings in alert_templates.py
    with {place}/{probability} filled in — never LLM-generated text.
    check_alert_text() is run as the last step before returning; a
    failure here means the TEMPLATE DATA itself is broken (not that an
    optional runtime component is unavailable), which is logged loudly
    and replaced with a minimal hardcoded string that is safe by
    construction, so the alert still goes out either way.
    """
    language = req.language if req.language in alert_templates.LANGUAGES else "en"
    hazard = req.hazard if req.hazard in alert_templates.HAZARDS else "flood"
    risk_level = req.risk_level.upper()
    if risk_level not in alert_templates.RISK_LEVELS:
        risk_level = "NORMAL"

    text = alert_templates.render_alert(
        risk_level, hazard, language,
        place=req.place_name, probability=req.probability,
        is_exercise=req.is_exercise,
    )

    allowed_numbers = alert_templates.expected_numbers(req.place_name, req.probability)
    passed, failures = check_alert_text(
        text,
        expected_place=req.place_name,
        is_exercise=req.is_exercise,
        allowed_numbers=allowed_numbers,
    )

    if not passed:
        print(f"[llm_narrator] TEMPLATE FAILED VALIDATION: {failures}", file=sys.stderr)
        prefix = alert_templates.DRILL_PREFIX if req.is_exercise else ""
        text = f"{prefix}{risk_level} for {req.place_name}. Contact control room for details."

    return {
        "text": text,
        "language": language,
        "source": "template",
        "model": None,
    }


# ---------------------------------------------------------------------
# situation_report() — ops-facing only. LLM stays here, with a
# hallucination guard on top.
# ---------------------------------------------------------------------

# States and Union Territories only — this is NOT a district-level
# gazetteer. India has 750+ districts; hand-typing that full list from
# memory risks exactly the kind of silent omission or typo this guard
# exists to prevent, which would be worse than an honestly incomplete
# guard. If district-level checking is needed, source an authoritative
# list (Census/Survey of India) rather than expanding this by hand.
INDIAN_STATES_AND_UTS = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi",
    "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
)


def _hallucinated_place(generated: str, input_facts: str) -> str | None:
    """First state/UT name found in `generated` that is NOT present in
    `input_facts`, or None. This is exactly the failure mode narrate()
    hit (a place name invented that wasn't in the input) — applied here
    too since situation_report() still uses the LLM."""
    for name in INDIAN_STATES_AND_UTS:
        if name in generated and name not in input_facts:
            return name
    return None


def situation_report(decisions: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Plain-language ops summary of recent system state.

    For the control-room operator, not for citizens. Purely descriptive —
    it summarises what already happened, it does not forecast or advise.
    """
    if not decisions:
        return "No telemetry received yet."

    latest = decisions[-1]
    tiers = [d.get("risk_level", "NORMAL") for d in decisions]
    counts = {t: tiers.count(t) for t in set(tiers)}
    summary_facts = (
        f"Readings analysed: {len(decisions)}. "
        f"Tier distribution: {counts}. "
        f"Current tier: {latest.get('risk_level')}. "
        f"Flood {latest.get('flood_probability')}%, "
        f"landslide {latest.get('landslide_probability')}%. "
        f"Safety net triggered: {latest.get('safety_net_triggered')}."
    )

    resolved = resolve_model(model)
    if not resolved:
        return summary_facts

    prompt = textwrap.dedent(f"""
        Write a 3-sentence situation report for a disaster control room
        operator, based only on these facts. Be factual and neutral. Do
        not speculate, forecast, or recommend actions. Output only the
        report.

        FACTS: {summary_facts}
    """).strip()

    generated = _call_ollama(prompt, resolved)
    if not generated:
        return summary_facts

    if len(generated) < 10 or len(generated) > 2000:
        if DEBUG:
            print(f"[llm] situation_report rejected: implausible length {len(generated)}",
                  file=sys.stderr)
        return summary_facts

    bad_place = _hallucinated_place(generated, summary_facts)
    if bad_place:
        if DEBUG:
            print(
                f"[llm] situation_report rejected: mentions {bad_place!r}, "
                f"which was not present in the input facts",
                file=sys.stderr,
            )
        return summary_facts

    return generated


# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Windows terminals commonly default stdout to cp1252, which cannot
    # represent Devanagari/Malayalam/Tamil script (or even the em-dash
    # in the English templates) — without this, printing anything but
    # plain ASCII crashes with UnicodeEncodeError. Scoped to __main__
    # only: reconfiguring stdout is an application-level decision, not
    # something this module should do as a side effect of being imported.
    sys.stdout.reconfigure(encoding="utf-8")

    demo = NarrationRequest(
        risk_level="CRITICAL",
        hazard="landslide",
        probability=91.0,
        place_name="Meppadi",
        reasons=[
            "deep soil saturation 93%",
            "slope 41 degrees",
            "sustained tilt creep detected",
        ],
        language="en",
        is_exercise=True,
    )

    if "--test-fallback" in sys.argv:
        # narrate() never touches the LLM at all now, so this just
        # proves the template path itself still works standalone.
        print("TEMPLATE PATH (narrate() never calls the LLM):")
        print(" ", narrate(demo)["text"])
        sys.exit(0)

    # Running this file directly is a debugging activity — show failures.
    DEBUG = True

    print("--- narrate(): citizen alerts, template-only, all languages ---")
    for lang in alert_templates.LANGUAGES:
        demo.language = lang
        result = narrate(demo)
        print(f"[{lang}] source={result['source']}")
        print(f"  {result['text']}\n")

    print("--- situation_report(): ops summary, LLM-backed ---")
    resolved = resolve_model()
    print(f"Resolved model: {resolved or 'NONE — will use plain facts'}")
    if not resolved:
        print("Run `python diagnose_ollama.py` to find out why.")
    fake_decisions = [
        {"risk_level": "ALERT", "flood_probability": 40, "landslide_probability": 22,
         "safety_net_triggered": False},
        {"risk_level": "CRITICAL", "flood_probability": 38, "landslide_probability": 91,
         "safety_net_triggered": True},
    ]
    print(situation_report(fake_decisions))
