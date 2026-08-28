"""
GiriKavach — Pre-written, human-verifiable citizen alert templates
======================================================================

WHY THIS FILE EXISTS
Real testing of the LLM-generated citizen alert path (llm_narrator.py,
pre-refactor) produced, in a single test run:
  - A hallucinated place name in Hindi output ("Uttarakhand" instead of
    the actual place, Meppadi, Kerala — ~2000km wrong)
  - The "[DRILL — NOT A REAL EMERGENCY]" prefix silently dropped
  - Raw sensor facts leaked verbatim despite an explicit prompt
    instruction not to
  - Garbled, misspelled Malayalam output

None of that is acceptable on a life-safety message path. The fix is
not a better prompt — it's removing the LLM from this path entirely.
Every string a citizen can receive is written here, ahead of time, by
a human, and only ever has {place} and {probability} substituted in at
runtime. Nothing is generated, translated, or rephrased at send time.

WHAT "VERIFIED" MEANS HERE
The English templates are the ones actually used to run this project's
existing demo/tests and are safe to treat as reviewed. The hi/ml/ta
templates are good-faith translations done for this pass, matching the
English meaning and structure — but they have NOT been checked by a
native speaker of each language, which matters enormously for a
life-safety message. Each non-English template is marked with a
TODO(i18n) comment for exactly that reason. Do not deploy to a
Hindi/Malayalam/Tamil-speaking area before that review happens.

WHY {place} IS NEVER TRANSLATED OR TRANSLITERATED
Even in the native-script templates, {place} is filled in with the
place name exactly as given (e.g. "Meppadi" stays in Latin script
inside a Malayalam sentence). This is deliberate, not an oversight:
transliterating place names is exactly the kind of step that produced
the original hallucination bug. Leaving it untouched also means
validate_alert.py's "expected place name appears verbatim" check works
identically regardless of language.

WHY THE DRILL PREFIX IS A SINGLE FIXED STRING, NOT TRANSLATED PER-LANGUAGE
The original bug where this prefix silently vanished happened in an
LLM rephrasing step that no longer exists — but even with that step
gone, translating "[DRILL — NOT A REAL EMERGENCY]" into four languages
would mean four independent strings that could each individually go
stale, get mistyped, or be the one one nobody reviewed. Keeping it as
one fixed, untranslated, always-identical string removes that risk
from the single most safety-critical span of text in the whole
message, and makes validate_alert.py's check a single exact substring
match regardless of which language the rest of the message is in.
"""

from __future__ import annotations

import re

DRILL_PREFIX = "[DRILL — NOT A REAL EMERGENCY] "

RISK_LEVELS = ("NORMAL", "WATCH", "ALERT", "CRITICAL")
HAZARDS = ("flood", "landslide")
LANGUAGES = ("en", "hi", "ml", "ta")


ALERT_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "CRITICAL": {
        "flood": {
            "en": (
                "URGENT — Flash flood danger at {place} ({probability:.0f}% "
                "probability). Move to high ground away from the river NOW. "
                "Do not wait. Do not cross flowing water."
            ),
            # TODO(i18n): needs native Hindi speaker verification before deployment
            "hi": (
                "अत्यावश्यक — {place} में अचानक बाढ़ का खतरा ({probability:.0f}% "
                "संभावना)। तुरंत नदी से दूर ऊंचे स्थान पर जाएं। इंतजार न करें। "
                "बहते पानी को पार न करें।"
            ),
            # TODO(i18n): needs native Malayalam speaker verification before deployment
            "ml": (
                "അടിയന്തിരം — {place}-ൽ പെട്ടെന്നുള്ള വെള്ളപ്പൊക്ക ഭീഷണി "
                "({probability:.0f}% സാധ്യത). ഉടൻ നദിയിൽ നിന്ന് അകന്ന് ഉയർന്ന "
                "സ്ഥലത്തേക്ക് പോകുക. കാത്തിരിക്കരുത്. ഒഴുകുന്ന വെള്ളം "
                "മുറിച്ചുകടക്കരുത്."
            ),
            # TODO(i18n): needs native Tamil speaker verification before deployment
            "ta": (
                "அவசரம் — {place}-இல் திடீர் வெள்ள அபாயம் ({probability:.0f}% "
                "வாய்ப்பு). உடனே ஆற்றிலிருந்து விலகி உயரமான இடத்திற்குச் "
                "செல்லவும். காத்திருக்க வேண்டாம். ஓடும் நீரைக் கடக்க வேண்டாம்."
            ),
        },
        "landslide": {
            "en": (
                "URGENT — Landslide danger at {place} ({probability:.0f}% "
                "probability). Move away from the slope to safe ground NOW. "
                "Do not wait. Do not shelter above or below a steep slope."
            ),
            # TODO(i18n): needs native Hindi speaker verification before deployment
            "hi": (
                "अत्यावश्यक — {place} में भूस्खलन का खतरा ({probability:.0f}% "
                "संभावना)। तुरंत ढलान से दूर सुरक्षित स्थान पर जाएं। इंतजार न "
                "करें। खड़ी ढलान के ऊपर या नीचे शरण न लें।"
            ),
            # TODO(i18n): needs native Malayalam speaker verification before deployment
            "ml": (
                "അടിയന്തിരം — {place}-ൽ മണ്ണിടിച്ചിൽ ഭീഷണി ({probability:.0f}% "
                "സാധ്യത). ഉടൻ ചെരിവിൽ നിന്ന് അകന്ന് സുരക്ഷിത സ്ഥലത്തേക്ക് "
                "പോകുക. കാത്തിരിക്കരുത്. കുത്തനെയുള്ള ചെരിവിന് മുകളിലോ "
                "താഴെയോ അഭയം തേടരുത്."
            ),
            # TODO(i18n): needs native Tamil speaker verification before deployment
            "ta": (
                "அவசரம் — {place}-இல் நிலச்சரிவு அபாயம் ({probability:.0f}% "
                "வாய்ப்பு). உடனே சரிவிலிருந்து விலகி பாதுகாப்பான "
                "இடத்திற்குச் செல்லவும். காத்திருக்க வேண்டாம். செங்குத்தான "
                "சரிவின் மேலோ கீழோ தங்காதீர்கள்."
            ),
        },
    },
    "ALERT": {
        "flood": {
            "en": (
                "WARNING — Flash flood risk at {place} ({probability:.0f}% "
                "probability). Prepare to move to higher ground. Stay away "
                "from the river channel."
            ),
            # TODO(i18n): needs native Hindi speaker verification before deployment
            "hi": (
                "चेतावनी — {place} में अचानक बाढ़ का जोखिम ({probability:.0f}% "
                "संभावना)। ऊंचे स्थान पर जाने की तैयारी करें। नदी के किनारे "
                "से दूर रहें।"
            ),
            # TODO(i18n): needs native Malayalam speaker verification before deployment
            "ml": (
                "മുന്നറിയിപ്പ് — {place}-ൽ പെട്ടെന്നുള്ള വെള്ളപ്പൊക്ക സാധ്യത "
                "({probability:.0f}%). ഉയർന്ന സ്ഥലത്തേക്ക് മാറാൻ "
                "തയ്യാറാകുക. നദിയുടെ സമീപത്ത് നിന്ന് അകന്നു നിൽക്കുക."
            ),
            # TODO(i18n): needs native Tamil speaker verification before deployment
            "ta": (
                "எச்சரிக்கை — {place}-இல் திடீர் வெள்ள ஆபத்து "
                "({probability:.0f}% வாய்ப்பு). உயரமான இடத்திற்குச் செல்ல "
                "தயாராகுங்கள். ஆற்றுப் பகுதியிலிருந்து விலகி இருங்கள்."
            ),
        },
        "landslide": {
            "en": (
                "WARNING — Landslide risk at {place} ({probability:.0f}% "
                "probability). Prepare to move away from slope bases. Watch "
                "for cracks in ground or walls."
            ),
            # TODO(i18n): needs native Hindi speaker verification before deployment
            "hi": (
                "चेतावनी — {place} में भूस्खलन का जोखिम ({probability:.0f}% "
                "संभावना)। ढलान के आधार से दूर जाने की तैयारी करें। जमीन या "
                "दीवारों में दरारों पर नज़र रखें।"
            ),
            # TODO(i18n): needs native Malayalam speaker verification before deployment
            "ml": (
                "മുന്നറിയിപ്പ് — {place}-ൽ മണ്ണിടിച്ചിൽ സാധ്യത "
                "({probability:.0f}%). ചെരിവിന്റെ അടിഭാഗത്ത് നിന്ന് മാറാൻ "
                "തയ്യാറാകുക. നിലത്തോ ചുവരിലോ വിള്ളലുകൾ ശ്രദ്ധിക്കുക."
            ),
            # TODO(i18n): needs native Tamil speaker verification before deployment
            "ta": (
                "எச்சரிக்கை — {place}-இல் நிலச்சரிவு ஆபத்து "
                "({probability:.0f}% வாய்ப்பு). சரிவின் அடிவாரத்திலிருந்து "
                "விலக தயாராகுங்கள். நிலத்திலோ சுவரிலோ விரிசல்களை "
                "கவனியுங்கள்."
            ),
        },
    },
    "WATCH": {
        "flood": {
            "en": (
                "ADVISORY — Flash flood conditions at {place} are being "
                "monitored ({probability:.0f}% probability). Stay alert and "
                "avoid the river channel."
            ),
            # TODO(i18n): needs native Hindi speaker verification before deployment
            "hi": (
                "सलाह — {place} में अचानक बाढ़ की स्थिति पर नज़र रखी जा रही "
                "है ({probability:.0f}% संभावना)। सतर्क रहें और नदी के "
                "किनारे से दूर रहें।"
            ),
            # TODO(i18n): needs native Malayalam speaker verification before deployment
            "ml": (
                "അറിയിപ്പ് — {place}-ലെ വെള്ളപ്പൊക്ക സാഹചര്യം "
                "നിരീക്ഷിച്ചുവരുന്നു ({probability:.0f}% സാധ്യത). ജാഗ്രത "
                "പാലിക്കുക, നദിയുടെ സമീപത്ത് നിന്ന് അകന്നു നിൽക്കുക."
            ),
            # TODO(i18n): needs native Tamil speaker verification before deployment
            "ta": (
                "அறிவிப்பு — {place}-இல் திடீர் வெள்ள நிலைமைகள் "
                "கண்காணிக்கப்படுகின்றன ({probability:.0f}% வாய்ப்பு). "
                "விழிப்புடன் இருங்கள், ஆற்றுப் பகுதியிலிருந்து விலகி "
                "இருங்கள்."
            ),
        },
        "landslide": {
            "en": (
                "ADVISORY — Landslide conditions at {place} are being "
                "monitored ({probability:.0f}% probability). Stay alert and "
                "avoid steep slopes."
            ),
            # TODO(i18n): needs native Hindi speaker verification before deployment
            "hi": (
                "सलाह — {place} में भूस्खलन की स्थिति पर नज़र रखी जा रही है "
                "({probability:.0f}% संभावना)। सतर्क रहें और खड़ी ढलानों से "
                "दूर रहें।"
            ),
            # TODO(i18n): needs native Malayalam speaker verification before deployment
            "ml": (
                "അറിയിപ്പ് — {place}-ലെ മണ്ണിടിച്ചിൽ സാഹചര്യം "
                "നിരീക്ഷിച്ചുവരുന്നു ({probability:.0f}% സാധ്യത). ജാഗ്രത "
                "പാലിക്കുക, കുത്തനെയുള്ള ചെരിവുകളിൽ നിന്ന് അകന്നു നിൽക്കുക."
            ),
            # TODO(i18n): needs native Tamil speaker verification before deployment
            "ta": (
                "அறிவிப்பு — {place}-இல் நிலச்சரிவு நிலைமைகள் "
                "கண்காணிக்கப்படுகின்றன ({probability:.0f}% வாய்ப்பு). "
                "விழிப்புடன் இருங்கள், செங்குத்தான சரிவுகளிலிருந்து விலகி "
                "இருங்கள்."
            ),
        },
    },
    "NORMAL": {
        "flood": {
            "en": (
                "No flash flood hazard detected at {place}. Sensors "
                "operating normally ({probability:.0f}% probability)."
            ),
            # TODO(i18n): needs native Hindi speaker verification before deployment
            "hi": (
                "{place} में कोई अचानक बाढ़ का खतरा नहीं मिला। सेंसर सामान्य "
                "रूप से काम कर रहे हैं ({probability:.0f}% संभावना)।"
            ),
            # TODO(i18n): needs native Malayalam speaker verification before deployment
            "ml": (
                "{place}-ൽ വെള്ളപ്പൊക്ക ഭീഷണിയൊന്നും കണ്ടെത്തിയിട്ടില്ല. "
                "സെൻസറുകൾ സാധാരണ നിലയിൽ പ്രവർത്തിക്കുന്നു ({probability:.0f}% "
                "സാധ്യത)."
            ),
            # TODO(i18n): needs native Tamil speaker verification before deployment
            "ta": (
                "{place}-இல் திடீர் வெள்ள ஆபத்து எதுவும் இல்லை. உணரிகள் "
                "இயல்பாக செயல்படுகின்றன ({probability:.0f}% வாய்ப்பு)."
            ),
        },
        "landslide": {
            "en": (
                "No landslide hazard detected at {place}. Sensors operating "
                "normally ({probability:.0f}% probability)."
            ),
            # TODO(i18n): needs native Hindi speaker verification before deployment
            "hi": (
                "{place} में कोई भूस्खलन का खतरा नहीं मिला। सेंसर सामान्य "
                "रूप से काम कर रहे हैं ({probability:.0f}% संभावना)।"
            ),
            # TODO(i18n): needs native Malayalam speaker verification before deployment
            "ml": (
                "{place}-ൽ മണ്ണിടിച്ചിൽ ഭീഷണിയൊന്നും കണ്ടെത്തിയിട്ടില്ല. "
                "സെൻസറുകൾ സാധാരണ നിലയിൽ പ്രവർത്തിക്കുന്നു ({probability:.0f}% "
                "സാധ്യത)."
            ),
            # TODO(i18n): needs native Tamil speaker verification before deployment
            "ta": (
                "{place}-இல் நிலச்சரிவு ஆபத்து எதுவும் இல்லை. உணரிகள் "
                "இயல்பாக செயல்படுகின்றன ({probability:.0f}% வாய்ப்பு)."
            ),
        },
    },
}


def get_template(risk_level: str, hazard: str, language: str) -> str:
    """Look up one template. Raises KeyError with a clear message rather
    than silently guessing if the combination doesn't exist — a missing
    template should fail loudly at the call site, not send blank text."""
    try:
        return ALERT_TEMPLATES[risk_level.upper()][hazard.lower()][language]
    except KeyError as e:
        raise KeyError(
            f"No template for risk_level={risk_level!r} hazard={hazard!r} "
            f"language={language!r}. risk_levels={RISK_LEVELS} "
            f"hazards={HAZARDS} languages={LANGUAGES}"
        ) from e


def render_alert(
    risk_level: str,
    hazard: str,
    language: str,
    place: str,
    probability: float,
    is_exercise: bool = True,
) -> str:
    """Fill a template's {place}/{probability} slots and prepend the
    drill prefix if applicable. This is the ONLY function that fills
    placeholders — keeping it in one place means narrate() and the
    tests are always formatting identically."""
    template = get_template(risk_level, hazard, language)
    text = template.format(place=place, probability=probability)
    if is_exercise:
        text = DRILL_PREFIX + text
    return text


def expected_numbers(place: str, probability: float) -> set[str]:
    """Digit-sequences legitimately allowed to appear in a rendered
    alert for these inputs: whatever {probability} formats to (matching
    the templates' own ":.0f" format spec), plus any digits already
    inside the place name itself. Anything else appearing in the
    rendered text was not in an input slot."""
    numbers = set(re.findall(r"\d+", place))
    numbers.add(f"{probability:.0f}")
    return numbers
