"""Thin, fail-safe wrapper around the Gemini API.

The single public entry point, :func:`ask_gemini_json`, NEVER raises for
network/API/parsing problems: it returns ``fallback_data`` instead, so the app
keeps working offline or without an API key.

Configuration (environment variables):
    GEMINI_API_KEY (or GOOGLE_API_KEY)  - enables real LLM calls
    GEMINI_MODEL                        - overrides the default model name
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
REQUEST_TIMEOUT_MS = 60_000

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

_JSON_INSTRUCTIONS = (
    "\n\nRespond with a single valid JSON object only. "
    "Do not wrap it in markdown code fences and do not add any commentary."
)


def get_api_key() -> str | None:
    for env_var in API_KEY_ENV_VARS:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    return None


def is_llm_available() -> bool:
    return get_api_key() is not None


def get_model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def extract_json_text(raw: str) -> str:
    """Strip markdown fences / surrounding chatter and return the JSON object text."""
    match = _CODE_FENCE_RE.match(raw)
    text = match.group(1) if match else raw.strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    # Imported lazily so the app (and tests) work even if the SDK is missing.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4,
        ),
    )
    return response.text or ""


def ask_gemini_json(
    prompt: str,
    fallback_data: dict,
    *,
    response_model: type[BaseModel] | None = None,
    model: str | None = None,
) -> dict:
    """Ask Gemini for a JSON object; return ``fallback_data`` on any failure.

    Schema enforcement:
      * If ``response_model`` is given, the response must validate against it.
      * Otherwise the response must contain every top-level key of ``fallback_data``.

    A deep copy of ``fallback_data`` is returned so callers can mutate the
    result without corrupting shared defaults.
    """
    api_key = get_api_key()
    if api_key is None:
        logger.info("No Gemini API key configured; using fallback data.")
        return copy.deepcopy(fallback_data)

    try:
        raw_text = _call_gemini(prompt + _JSON_INSTRUCTIONS, api_key, model or get_model_name())
        data = json.loads(extract_json_text(raw_text))
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")

        if response_model is not None:
            return response_model.model_validate(data).model_dump(mode="json")

        missing_keys = set(fallback_data) - set(data)
        if missing_keys:
            raise ValueError(f"response is missing keys: {sorted(missing_keys)}")
        return data
    except (ValidationError, ValueError) as exc:
        logger.warning("Gemini returned unusable JSON (%s); using fallback data.", exc)
    except Exception as exc:  # network errors, quota, auth, missing SDK, ...
        logger.warning("Gemini call failed (%s: %s); using fallback data.", type(exc).__name__, exc)
    return copy.deepcopy(fallback_data)
