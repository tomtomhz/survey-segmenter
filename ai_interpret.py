"""ai_interpret.py — the optional "ask Claude about your segments" layer for the Survey Segmenter.

After the statistics tool has found the customer groups, this module sends the AGGREGATE report
(segment sizes, the preferences that define each group, the confidence rating, and the demographic
profile) to Claude and gets back a plain-language interpretation, then answers follow-up questions
in a chat. The user can "dump the results in and have Claude interpret them", exactly as asked.

Design decisions (deliberate):

* Uses the OFFICIAL Anthropic Python SDK (`anthropic`) and the model `claude-opus-5`. This is an
  OPTIONAL add-on: if the SDK is not installed, or no API key is configured, the whole
  segmentation tool still works — the chat panel just explains what to add. Nothing here is
  imported when the core tool runs.

* Privacy by construction. Only the aggregate report TEXT leaves the computer — never a single
  respondent's row. The report the tool produces is already all-aggregate (sizes, mean scores,
  stability numbers, demographic percentages); that is what we send.

* The user brings their OWN Anthropic API key (from console.anthropic.com), read from the
  ANTHROPIC_API_KEY environment variable or from a small local settings file they fill in through
  the app. The key never leaves their machine except to talk to Anthropic under their own account.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# claude-opus-5 is the current, most capable Opus model — the right default for turning a
# statistical readout into business recommendations. (The user can point at another model by
# setting SURVEY_SEGMENTER_MODEL, but the default needs no configuration.)
MODEL = os.environ.get("SURVEY_SEGMENTER_MODEL", "claude-opus-5")

_CONFIG_DIR = Path.home() / ".survey_segmenter"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

# Room for adaptive thinking PLUS a full written answer (thinking and text share this budget on
# Opus 5). Generous so a rich first interpretation never truncates; streaming avoids timeouts.
_MAX_TOKENS = 6000

# Opus 5's safety classifiers can decline a request outright — the call succeeds, but comes back
# with stop_reason "refusal" and no answer. Segment reports are dry marketing statistics, so this
# should be vanishingly rare, but a false positive would look to the user like the tool is broken.
# Server-side fallbacks re-run the declined request on another model inside the same call, routed
# by refusal category, so the user gets an answer instead of an apology. "default" lets Anthropic
# pick the substitute rather than pinning a model we would then have to maintain.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = (
    "You are a segmentation strategist embedded with a marketing, business-development, or growth "
    "team. You are handed the AGGREGATE output of a rigorous customer-segmentation study: segment "
    "sizes, the preferences that most define each segment, stability/confidence diagnostics, and "
    "(when present) demographic profiles and a population projection. A statistician has already "
    "done the maths — the numbers are final and correct. NEVER recompute, second-guess, or invent "
    "figures; your job is to translate the result into decisions.\n\n"
    "In your FIRST reply, cover, in this order:\n"
    "1. A one-paragraph plain-English readout: how many groups, and what fundamentally separates "
    "them.\n"
    "2. Each segment as a short, vivid NAME plus a one- to two-sentence persona, with its size.\n"
    "3. Which segment(s) to prioritise and why — grounded in size, distinctiveness, and the "
    "confidence rating.\n"
    "4. Two or three concrete go-to-market or product actions for each priority segment.\n"
    "5. An honest 'how much to trust this' note based on the confidence light and stability "
    "numbers. If the report is amber/red or the segments look unstable, LEAD with that caveat and "
    "do not over-promise.\n\n"
    "Be concrete and useful, never academic. Use the actual sizes (and the population projection "
    "if given). Never state a number that is not in the report. For follow-up questions, answer "
    "directly and briefly, always tying advice back to what the data actually supports; if the "
    "data cannot answer, say so.\n\n"
    "Formatting: use Markdown that renders cleanly — '## ' and '### ' headings, '- ' bullet lists, "
    "and '**bold**' for segment names and key terms. Do NOT use numbered lists or Markdown tables. "
    "Keep paragraphs short and skimmable."
)

_INITIAL_INSTRUCTION = ("Read this customer-segmentation report and give the team your full "
                        "interpretation and recommendations.")


NAMING_PROMPT = (
    "Below is a customer-segmentation report. Give each group a short, memorable name a marketing "
    "team would actually use — two or three words, in the customers' world, not the statistics' "
    "(for example 'Privacy-First Lurkers' or 'Real-Life Connectors', never 'Segment 2' or a list of "
    "question codes). Base each name on what genuinely distinguishes that group in the report. "
    "Return exactly {k} names, in segment order starting from Segment 0."
)


def suggest_names(report_markdown: str, k: int, api_key: "str | None" = None, model: str = MODEL):
    """Ask Claude for a short, human name per segment. Uses structured output so we get a clean
    list back instead of having to scrape prose — names feed the exports, so they must be reliable.
    Returns a list of exactly `k` strings."""
    if not have_sdk():
        raise AIError("The AI add-on isn't installed, so I cannot suggest names. You can still "
                      "type your own.", kind="nosdk")
    key = api_key or load_api_key()
    if not key:
        raise AIError("Add your Anthropic API key in Settings to have Claude name the groups. "
                      "You can still type your own names.", kind="nokey")
    import anthropic
    schema = {"type": "object",
              "properties": {"names": {"type": "array", "items": {"type": "string"}}},
              "required": ["names"], "additionalProperties": False}
    client = anthropic.Anthropic(api_key=key)
    try:
        resp = client.messages.create(
            model=model, max_tokens=1200,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": NAMING_PROMPT.format(k=k) + "\n\n"
                       + report_markdown}],
        )
    except anthropic.AuthenticationError:
        raise AIError("Your Anthropic API key was not accepted. Check it in Settings.", kind="auth")
    except anthropic.APIError as e:
        raise AIError(f"Claude could not be reached right now: {e}", kind="error")
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    try:
        names = [str(n).strip() for n in json.loads(text)["names"] if str(n).strip()]
    except Exception:
        raise AIError("Claude's reply could not be read as a list of names. Try again, or type "
                      "your own.", kind="error")
    if len(names) < k:                       # never leave a group unnamed
        names += [f"Group {i}" for i in range(len(names), k)]
    return names[:k]


class AIError(Exception):
    """A problem we can explain to a non-technical user in plain language."""

    def __init__(self, message, kind="error"):
        super().__init__(message)
        self.kind = kind          # "nosdk" | "nokey" | "auth" | "error" — lets the UI react


def have_sdk() -> bool:
    """Is the Anthropic SDK available? Checked without paying the import cost."""
    import importlib.util
    try:
        return importlib.util.find_spec("anthropic") is not None
    except Exception:
        return False


def _read_config() -> dict:
    try:
        data = json.loads(_CONFIG_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_api_key() -> "str | None":
    """The active key: the ANTHROPIC_API_KEY env var wins (matching the SDK), else the app's own
    settings file. Returns None if neither is set."""
    env = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if env:
        return env
    key = _read_config().get("api_key")
    return key.strip() if isinstance(key, str) and key.strip() else None


def key_source() -> "str | None":
    if (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return "the ANTHROPIC_API_KEY environment variable"
    if _read_config().get("api_key"):
        return "this app's Settings"
    return None


def save_api_key(key: str) -> None:
    """Store the user's key in ~/.survey_segmenter/config.json (readable only by them). It stays on
    this computer and is used only to call Claude under the user's own Anthropic account."""
    key = (key or "").strip()
    if not key:
        raise AIError("Please paste your Anthropic API key first.", kind="nokey")
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _read_config()
    cfg["api_key"] = key
    _CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    try:
        os.chmod(_CONFIG_FILE, 0o600)     # owner read/write only
    except Exception:
        pass


def clear_api_key() -> None:
    cfg = _read_config()
    if "api_key" in cfg:
        cfg.pop("api_key", None)
        try:
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            _CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        except Exception:
            pass


def status() -> dict:
    """What the UI needs to decide what to show: is the SDK there, is a key set, and from where.
    Never returns the key itself."""
    env_key = bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())
    return {"sdk_installed": have_sdk(),
            "configured": load_api_key() is not None,
            "source": key_source(),
            "env_key": env_key,          # if True, the key is env-set and can't be changed in-app
            "model": MODEL}


def _first_turn_content(report_markdown: str, question: "str | None") -> str:
    ask = (question or "").strip() or _INITIAL_INSTRUCTION
    return (ask + "\n\nHere is the full segmentation report. Every figure in it is final.\n\n"
            "----- BEGIN REPORT -----\n" + report_markdown + "\n----- END REPORT -----")


def build_messages(report_markdown: str, question: "str | None", history: "list | None") -> list:
    """Pure helper (no network): produce the Anthropic `messages` list for the next turn.

    On the FIRST turn (`history` empty/None) the report is embedded into the user message and
    `question` may be None (auto-interpretation). On later turns the running `history` already
    carries the report, so we just append the new question. Returned as a fresh list."""
    if history:
        msgs = list(history)
        msgs.append({"role": "user", "content": (question or "Please continue.").strip()})
        return msgs
    return [{"role": "user", "content": _first_turn_content(report_markdown, question)}]


def chat_once(history: "list | None", report_markdown: str, question: "str | None" = None,
              api_key: "str | None" = None, model: str = MODEL):
    """Advance the conversation by one turn and call Claude.

    `history` is the running message list (empty on the first turn). Returns
    (reply_text, new_history). Raises AIError with a `.kind` the UI can act on when the SDK is
    missing, no key is set, or the key is rejected."""
    if not have_sdk():
        raise AIError("The AI add-on isn't installed. Install it with:  pip install anthropic  "
                      "(or rebuild the app), then reopen this page.", kind="nosdk")
    key = (api_key or load_api_key())
    if not key:
        raise AIError("Add your Anthropic API key in Settings to have Claude interpret your "
                      "results.", kind="nokey")

    import anthropic

    messages = build_messages(report_markdown, question, history)
    client = anthropic.Anthropic(api_key=key)

    # Stream + get_final_message: robust against timeouts on long input/output, per the Anthropic SDK
    # guidance. Adaptive thinking (on by default on Opus 5) with medium effort is a good
    # quality/latency balance for an interpretation chat. If an older installed SDK does not accept
    # those tuning kwargs, fall back to a plain request so the chat still works.
    # Best request first, then progressively plainer ones. Each rung drops a capability that an
    # older SDK or an account without the beta might reject, so the chat still works everywhere
    # rather than failing for anyone not on the newest setup.
    _tuning = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}
    attempts = (
        ("beta", {**_tuning, "betas": [_FALLBACK_BETA], "fallbacks": "default"}),
        ("plain", _tuning),
        ("plain", {}),
    )
    final = None
    for endpoint, extra in attempts:
        try:
            api = client.beta.messages if endpoint == "beta" else client.messages
            with api.stream(model=model, max_tokens=_MAX_TOKENS,
                            system=SYSTEM_PROMPT, messages=messages, **extra) as stream:
                final = stream.get_final_message()
            break
        except (TypeError, AttributeError):
            final = None
            continue          # this SDK version does not know these kwargs; retry without them
        except anthropic.BadRequestError:
            # Most likely this account is not enabled for the fallbacks beta. That is a reason to
            # ask for less, not to break the feature — drop to the next rung.
            final = None
            continue
        except anthropic.AuthenticationError:
            raise AIError("Your Anthropic API key was not accepted. Check it in Settings "
                          "(it should start with 'sk-ant-').", kind="auth")
        except anthropic.APIError as e:
            raise AIError(f"Claude could not be reached right now: {e}. "
                          "Check your internet connection and try again.", kind="error")
    if final is None:
        raise AIError("The installed Anthropic SDK is too old for this app. Update it with:  "
                      "pip install -U anthropic", kind="error")

    stop = getattr(final, "stop_reason", None)
    if stop == "refusal":
        text = ("Claude declined to answer that one. Try rephrasing, or ask about a different "
                "aspect of the segments.")
    else:
        text = "".join(b.text for b in final.content
                       if getattr(b, "type", None) == "text").strip()
        if not text:
            # Thinking and the written answer share the token budget, so a very deep think can
            # leave nothing for the reply. Say so instead of showing a blank bubble.
            text = ("Claude spent its budget thinking and did not get to a written answer. "
                    "Please ask again, or ask a narrower question."
                    if stop == "max_tokens" else
                    "Claude returned an empty answer. Please try asking again.")
        elif stop == "max_tokens":
            # Never present a cut-off answer as if it were complete.
            text += "\n\n*(This answer was cut short at the length limit — ask me to continue.)*"

    new_history = list(messages)
    new_history.append({"role": "assistant", "content": text})
    return text, new_history
