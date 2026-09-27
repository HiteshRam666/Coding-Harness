"""providers.py — every vendor adapter in one file, behind one neutral format.

The concept, unchanged from the split version: the rest of the harness speaks
a small vendor-free dialect — user, assistant, tool — and this module is the
only translator between that dialect and each vendor's wire format.

Neutral message format (the only dialect anything outside this file speaks):
  {"role": "user",      "text": str}
  {"role": "assistant", "text": str, "tool_calls": [{"name", "args", "id"?, "signature"?}]}
  {"role": "tool",      "name": str, "text": str, "id"?: str}

`id` is a provider-issued call handle OpenAI and Anthropic need echoed back
verbatim on the matching tool result; `signature` is Gemini's opaque
thought-signature, round-tripped the same way. A provider that doesn't need
one simply ignores it.

`tools` is a list of {"schema": {"name", "description", "parameters"}} — one
shape handed to every vendor; each adapter reshapes it into that vendor's
wire format.

complete(model, system, messages, tools) returns:
  {"text": str,
   "tool_calls": [{"name": str, "args": dict, "id"?: str, "signature"?: str}],
   "usage": {"input": int, "output": int}}

Design rules every adapter follows:
  - One boundary per vendor. When a vendor changes their JSON, only that
    adapter's functions change.
  - Return plain dicts, never SDK/response objects: nothing above this file
    should learn a vendor's SDK.
  - Retry the transient, raise the permanent: a 429 is weather, a 400 is a bug.

Public surface: `resolve(spec)` -> (provider_name, model_name, complete_fn).
Everything else is grouped by vendor and safe to ignore unless you're adding
or debugging one.
"""

import json
import os
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Shared HTTP retry helper — one place for the weather-vs-bug judgment call,
# used by every adapter below instead of being copy-pasted three times.
# --------------------------------------------------------------------------

def _post_json(url, body, headers, retries=5, timeout=600):
    """POST JSON, return the parsed reply. Retries 429/5xx/dropped
    connections with exponential backoff; anything else is raised
    immediately, since retrying a malformed request never fixes it."""
    payload = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            detail = e.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt >= retries - 1:
                raise
            time.sleep(2 ** attempt * 2)


# --------------------------------------------------------------------------
# Gemini
#
# Quirk this section hides: Gemini has no "tool" role — a result goes back as
# a *user* turn carrying a functionResponse part — and Gemini 3 requires each
# functionCall's opaque thoughtSignature to be echoed back verbatim on the
# next turn, riding on that same part, or it rejects the exchange.
# --------------------------------------------------------------------------

_GEMINI_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"


def _gemini_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No Gemini key found. Set GEMINI_API_KEY.")
    return key


def _gemini_to_wire(messages):
    wire = []
    for m in messages:
        if m["role"] == "user":
            wire.append({"role": "user", "parts": [{"text": m.get("text", "")}]})
        elif m["role"] == "assistant":
            parts = [{"text": m["text"]}] if m.get("text") else []
            for call in m.get("tool_calls") or []:
                part = {"functionCall": {"name": call["name"], "args": call["args"]}}
                if call.get("signature"):
                    part["thoughtSignature"] = call["signature"]
                parts.append(part)
            wire.append({"role": "model", "parts": parts})
        elif m["role"] == "tool":
            wire.append({"role": "user", "parts": [{"functionResponse": {
                "name": m["name"], "response": {"result": m.get("text", "")}}}]})
    return wire


def _gemini_complete(model, system, messages, tools):
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": _gemini_to_wire(messages),
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 65536}}
    if tools:
        body["tools"] = [{"functionDeclarations": [t["schema"] for t in tools]}]
    headers = {"Content-Type": "application/json", "x-goog-api-key": _gemini_api_key()}
    data = _post_json(f"{_GEMINI_API_ROOT}/{model}:generateContent", body, headers)

    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
    text, calls = [], []
    for part in parts:
        if "text" in part and not part.get("thought"):
            text.append(part["text"])
        if "functionCall" in part:
            fc = part["functionCall"]
            calls.append({"name": fc.get("name", ""), "args": fc.get("args") or {},
                          "signature": part.get("thoughtSignature")})
    usage = data.get("usageMetadata") or {}
    return {"text": "".join(text).strip(), "tool_calls": calls,
            "usage": {"input": usage.get("promptTokenCount", 0),
                      "output": usage.get("candidatesTokenCount", 0)}}


# --------------------------------------------------------------------------
# OpenAI (Chat Completions)
#
# Quirk this section hides: OpenAI *does* have a "tool" role, but every tool
# call and its result are correlated by an `id` string the vendor issues on
# the call and expects back verbatim on the result — a plain id round-trip,
# no signature-on-the-part trick like Gemini.
# --------------------------------------------------------------------------

_OPENAI_API_ROOT = "https://api.openai.com/v1/chat/completions"


def _openai_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No OpenAI key found. Set OPENAI_API_KEY.")
    return key


def _openai_to_wire(system, messages):
    wire = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            wire.append({"role": "user", "content": m.get("text", "")})
        elif m["role"] == "assistant":
            entry = {"role": "assistant", "content": m.get("text") or None}
            calls = m.get("tool_calls") or []
            if calls:
                entry["tool_calls"] = [
                    {"id": c.get("id"), "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                    for c in calls]
            wire.append(entry)
        elif m["role"] == "tool":
            wire.append({"role": "tool", "tool_call_id": m.get("id"),
                        "content": m.get("text", "")})
    return wire


def _openai_complete(model, system, messages, tools):
    body = {"model": model, "messages": _openai_to_wire(system, messages),
            "temperature": 0.4}
    if tools:
        body["tools"] = [{"type": "function", "function": t["schema"]} for t in tools]
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {_openai_api_key()}"}
    data = _post_json(_OPENAI_API_ROOT, body, headers)

    choice = (data.get("choices") or [{}])[0].get("message") or {}
    text = choice.get("content") or ""
    calls = []
    for tc in choice.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": fn.get("name", ""), "args": args, "id": tc.get("id")})
    usage = data.get("usage") or {}
    return {"text": text.strip(), "tool_calls": calls,
            "usage": {"input": usage.get("prompt_tokens", 0),
                      "output": usage.get("completion_tokens", 0)}}


# --------------------------------------------------------------------------
# Anthropic (Messages API)
#
# Quirk this section hides: no "tool" role here either — a result goes back
# as a *user* turn carrying a tool_result content block, correlated to the
# call by the same `id` Anthropic issued on the tool_use block (same
# id-round-trip idea as OpenAI, different envelope).
# --------------------------------------------------------------------------

_ANTHROPIC_API_ROOT = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"


def _anthropic_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No Anthropic key found. Set ANTHROPIC_API_KEY.")
    return key


def _anthropic_to_tool(schema):
    """Same {name, description, parameters} shape every adapter is handed;
    Anthropic just calls the parameters key `input_schema`."""
    return {"name": schema["name"], "description": schema.get("description", ""),
            "input_schema": schema.get("parameters") or schema.get("input_schema") or {}}


def _anthropic_to_wire(messages):
    wire = []
    for m in messages:
        if m["role"] == "user":
            wire.append({"role": "user", "content": m.get("text", "")})
        elif m["role"] == "assistant":
            content = []
            if m.get("text"):
                content.append({"type": "text", "text": m["text"]})
            for c in m.get("tool_calls") or []:
                content.append({"type": "tool_use", "id": c.get("id"),
                                "name": c["name"], "input": c["args"]})
            wire.append({"role": "assistant", "content": content})
        elif m["role"] == "tool":
            wire.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": m.get("id"),
                 "content": m.get("text", "")}]})
    return wire


def _anthropic_complete(model, system, messages, tools):
    body = {"model": model, "system": system, "max_tokens": 8192,
            "temperature": 0.4, "messages": _anthropic_to_wire(messages)}
    if tools:
        body["tools"] = [_anthropic_to_tool(t["schema"]) for t in tools]
    headers = {"Content-Type": "application/json", "x-api-key": _anthropic_api_key(),
               "anthropic-version": _ANTHROPIC_API_VERSION}
    data = _post_json(_ANTHROPIC_API_ROOT, body, headers)

    text, calls = [], []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            text.append(block["text"])
        elif block.get("type") == "tool_use":
            calls.append({"name": block.get("name", ""), "args": block.get("input") or {},
                          "id": block.get("id")})
    usage = data.get("usage") or {}
    return {"text": "".join(text).strip(), "tool_calls": calls,
            "usage": {"input": usage.get("input_tokens", 0),
                      "output": usage.get("output_tokens", 0)}}


# --------------------------------------------------------------------------
# Registry — the one place that knows provider names exist. cli.py and the
# agent loop hold a "provider:model" string and never call an adapter above
# directly.
#
# Adding a fourth vendor: write its _xyz_complete(model, system, messages,
# tools) above, matching the same return shape, then add one line to
# _COMPLETERS. Nothing else changes.
# --------------------------------------------------------------------------

_COMPLETERS = {
    "anthropic": _anthropic_complete,
    "openai": _openai_complete,
    "gemini": _gemini_complete,
}

# Bare model names that don't need "provider:" spelled out, since the name
# alone is unambiguous. Extend freely; unknown names still require a prefix.
_MODEL_HINTS = {
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "claude": "anthropic",
    "gemini": "gemini",
}


def resolve(spec: str):
    """spec is "provider:model" (e.g. "openai:gpt-4o-mini") or, when the
    model name is unambiguous, just "model" (e.g. "gpt-4o-mini"). Returns
    (provider_name, model_name, complete_fn), where complete_fn(model,
    system, messages, tools) is one of the _*_complete functions above."""
    if ":" in spec:
        provider, model = spec.split(":", 1)
    else:
        provider, model = _guess_provider(spec), spec
    provider = provider.strip().lower()
    complete_fn = _COMPLETERS.get(provider)
    if complete_fn is None:
        known = ", ".join(sorted(_COMPLETERS))
        raise ValueError(f"Unknown provider '{provider}'. Known: {known}. "
                         f"Use provider:model, e.g. openai:{spec}.")
    return provider, model.strip(), complete_fn


def _guess_provider(model: str) -> str:
    lowered = model.lower()
    for hint, provider in _MODEL_HINTS.items():
        if lowered.startswith(hint):
            return provider
    raise ValueError(f"Can't tell which provider '{model}' belongs to — "
                     f"specify it as provider:model, e.g. anthropic:{model}.")