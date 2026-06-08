"""LLM provider abstraction.

One env var (LLM_PROVIDER=gemini|groq) switches backend. Every node calls
`invoke_structured` for typed extraction; tenacity wraps each call to absorb
transient 429s and structured-output parse glitches.

STREAMING SUPPORT
-----------------
When a caller (typically the Streamlit UI) wants live token visibility, it sets
the `current_on_token` context-variable before invoking a node. Every
`invoke_structured` call detects the ContextVar and switches from the buffered
`.with_structured_output(schema).invoke(...)` path to a streaming raw-JSON path
that emits chunks via the callback and parses the accumulated buffer at the end.
Node code is untouched — the ContextVar is the dependency injection seam.
"""
from __future__ import annotations

import json
import re
from contextvars import ContextVar
from typing import Callable, Optional, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    DEFAULT_TEMPERATURE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MAX_RETRIES,
)

T = TypeVar("T", bound=BaseModel)


# ContextVar set by the streaming layer. When None (default), invoke_structured
# uses the original buffered path.
current_on_token: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
    "current_on_token", default=None
)


def get_llm(temperature: float = DEFAULT_TEMPERATURE) -> BaseChatModel:
    """Return a chat model based on LLM_PROVIDER env var."""
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY missing. Set it in .env "
                "(free key: https://aistudio.google.com/apikey)"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=temperature,
            google_api_key=GEMINI_API_KEY,
        )

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY missing. Set it in .env "
                "(free key: https://console.groq.com/keys)"
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=temperature,
            groq_api_key=GROQ_API_KEY,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Use 'gemini' or 'groq'."
    )


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s).strip()


def _extract_json_blob(s: str) -> str:
    """Last-ditch: take the widest balanced { ... } slice."""
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in streamed buffer")
    return s[start : end + 1]


def _extract_delta(chunk) -> str:
    """Pull a displayable text delta from a streaming AIMessageChunk.

    For both Gemini and Groq, regular `.content` carries the JSON body when the
    model is asked for raw JSON. If the provider routes the response through
    tool_call_chunks instead (rare for our explicit JSON-only prompt), we fall
    back to that.
    """
    content = getattr(chunk, "content", "") or ""
    if content:
        return content
    tcc = getattr(chunk, "tool_call_chunks", None)
    if tcc:
        return "".join(part.get("args", "") or "" for part in tcc)
    return ""


def _stream_into_schema(
    llm: BaseChatModel,
    schema: Type[T],
    system_prompt: str,
    user_prompt: str,
    on_token: Callable[[str], None],
) -> T:
    """Stream a raw JSON response, push tokens to `on_token`, parse into schema.

    Adds a strict JSON-only suffix to the system prompt so the model emits
    parseable JSON without prose or fences. Falls back to a code-fence strip
    and a widest-balanced-brace slice if the buffer doesn't parse cleanly.
    """
    from src.prompts import JSON_ONLY_GUARD

    sys2 = system_prompt + JSON_ONLY_GUARD.format(
        json_schema=json.dumps(schema.model_json_schema(), indent=2)
    )

    # Reset signal so the UI can clear stale token buffer on retry.
    on_token("\n")

    buffer: list[str] = []
    for chunk in llm.stream(
        [SystemMessage(content=sys2), HumanMessage(content=user_prompt)]
    ):
        delta = _extract_delta(chunk)
        if delta:
            buffer.append(delta)
            on_token(delta)

    raw = _strip_code_fences("".join(buffer))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(_extract_json_blob(raw))
    return schema.model_validate(data)


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def invoke_structured(
    schema: Type[T],
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    on_token: Optional[Callable[[str], None]] = None,
) -> T:
    """Force LLM to return a validated instance of `schema`.

    If `on_token` is None, the active `current_on_token` ContextVar is used.
    When a callback is in play, switches to the streaming raw-JSON path so the
    caller can render tokens live. Otherwise uses the original buffered
    `with_structured_output` path.
    """
    on_token = on_token or current_on_token.get()
    llm = get_llm(temperature=temperature)

    if on_token is not None:
        return _stream_into_schema(llm, schema, system_prompt, user_prompt, on_token)

    structured_llm = llm.with_structured_output(schema)
    result = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    if isinstance(result, dict):
        result = schema(**result)
    return result


def invoke_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Plain-text call for the drafter node (markdown body generation)."""
    llm = get_llm(temperature=temperature)
    result = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return result.content if hasattr(result, "content") else str(result)
