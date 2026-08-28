"""
GiriKavach — Alert safety tests
==================================
Run:  python test_alert_safety.py

Proves three things, each corresponding directly to a real failure
this session's LLM testing produced:

  1. Every (risk_level, hazard, language) template renders and PASSES
     check_alert_text() — the templates are safe by construction, not
     by hoping the LLM behaves.
  2. Deliberately corrupted messages FAIL validation, each for the
     specific reason it should — proves check_alert_text() actually
     catches the failure modes seen (hallucinated place, dropped drill
     prefix, invented number), not just rubber-stamping everything.
  3. situation_report() degrades to plain facts, not silence or a
     crash, when the LLM is unreachable — tested by mocking
     resolve_model(), not by hoping the local Ollama happens to be off.
"""

import unittest
from unittest.mock import patch

import alert_templates
import llm_narrator
from alert_templates import ALERT_TEMPLATES, DRILL_PREFIX, render_alert
from validate_alert import check_alert_text


class TestEveryTemplatePassesValidation(unittest.TestCase):
    def test_every_combo_passes(self):
        failures_found = []
        for risk_level, by_hazard in ALERT_TEMPLATES.items():
            for hazard, by_lang in by_hazard.items():
                for lang in by_lang:
                    for is_exercise in (True, False):
                        text = render_alert(
                            risk_level, hazard, lang,
                            place="Meppadi", probability=87.0,
                            is_exercise=is_exercise,
                        )
                        ok, failures = check_alert_text(
                            text,
                            expected_place="Meppadi",
                            is_exercise=is_exercise,
                            allowed_numbers={"87"},
                        )
                        if not ok:
                            failures_found.append(
                                (risk_level, hazard, lang, is_exercise, failures)
                            )
        self.assertEqual(
            failures_found, [],
            f"{len(failures_found)} template combo(s) failed validation:\n"
            + "\n".join(str(f) for f in failures_found),
        )

    def test_covers_all_declared_dimensions(self):
        # Guards against a template silently missing from the dict —
        # the loop above only checks what's THERE, this checks nothing
        # declared in RISK_LEVELS/HAZARDS/LANGUAGES is absent.
        for risk_level in alert_templates.RISK_LEVELS:
            for hazard in alert_templates.HAZARDS:
                for lang in alert_templates.LANGUAGES:
                    self.assertIn(risk_level, ALERT_TEMPLATES)
                    self.assertIn(hazard, ALERT_TEMPLATES[risk_level])
                    self.assertIn(lang, ALERT_TEMPLATES[risk_level][hazard])


class TestCorruptedMessagesFail(unittest.TestCase):
    """Each test corrupts one real, rendered alert in exactly the way
    this session's actual LLM testing broke it, and proves
    check_alert_text() catches that specific failure."""

    def setUp(self):
        self.text = render_alert(
            "CRITICAL", "landslide", "en",
            place="Meppadi", probability=91.0, is_exercise=True,
        )

    def test_hallucinated_place_fails(self):
        # The real bug: "Uttarakhand" instead of "Meppadi".
        corrupted = self.text.replace("Meppadi", "Uttarakhand")
        ok, failures = check_alert_text(
            corrupted, expected_place="Meppadi", is_exercise=True,
            allowed_numbers={"91"},
        )
        self.assertFalse(ok)
        self.assertTrue(any("place" in f.lower() for f in failures), failures)

    def test_missing_drill_prefix_fails(self):
        corrupted = self.text.replace(DRILL_PREFIX, "")
        ok, failures = check_alert_text(
            corrupted, expected_place="Meppadi", is_exercise=True,
            allowed_numbers={"91"},
        )
        self.assertFalse(ok)
        self.assertTrue(any("drill" in f.lower() for f in failures), failures)

    def test_invented_number_fails(self):
        corrupted = self.text + " Evacuate within 15 minutes."
        ok, failures = check_alert_text(
            corrupted, expected_place="Meppadi", is_exercise=True,
            allowed_numbers={"91"},
        )
        self.assertFalse(ok)
        self.assertTrue(
            any("number" in f.lower() or "digit" in f.lower() for f in failures),
            failures,
        )

    def test_too_short_fails(self):
        ok, failures = check_alert_text(
            "short", expected_place="Meppadi", is_exercise=True, allowed_numbers=set(),
        )
        self.assertFalse(ok)
        self.assertTrue(any("length" in f.lower() for f in failures), failures)

    def test_uncorrupted_message_passes_as_control(self):
        # If this fails, the corruption tests above are meaningless —
        # confirms the baseline itself is valid before trusting the
        # "corrupting it breaks validation" claims.
        ok, failures = check_alert_text(
            self.text, expected_place="Meppadi", is_exercise=True,
            allowed_numbers={"91"},
        )
        self.assertTrue(ok, failures)


class TestSituationReportFallback(unittest.TestCase):
    def test_falls_back_to_plain_facts_when_llm_down(self):
        decisions = [{
            "risk_level": "ALERT",
            "flood_probability": 40,
            "landslide_probability": 22,
            "safety_net_triggered": False,
        }]
        with patch.object(llm_narrator, "resolve_model", return_value=None):
            report = llm_narrator.situation_report(decisions)
        # Must still contain the real facts — degraded, not silent.
        self.assertIn("40", report)
        self.assertIn("ALERT", report)

    def test_falls_back_on_hallucinated_place(self):
        # Simulates the LLM inventing a state name that was never in
        # the input facts — the guard from point 3, exercised directly.
        decisions = [{
            "risk_level": "WATCH",
            "flood_probability": 12,
            "landslide_probability": 8,
            "safety_net_triggered": False,
        }]
        with patch.object(llm_narrator, "resolve_model", return_value="fake-model"), \
             patch.object(
                 llm_narrator, "_call_ollama",
                 return_value="Conditions in Uttarakhand remain stable.",
             ):
            report = llm_narrator.situation_report(decisions)
        self.assertNotIn("Uttarakhand", report)
        self.assertIn("12", report)  # fell back to summary_facts

    def test_empty_decisions_does_not_crash(self):
        self.assertEqual(llm_narrator.situation_report([]), "No telemetry received yet.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
