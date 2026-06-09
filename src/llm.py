"""LLM provider abstraction.

LLM_PROVIDER env var (gemini | groq | openrouter) switches backend.
`invoke_structured` is the single typed entry-point used by every node;
tenacity wraps it to absorb 429s and structured-output parse glitches.

Streaming: when a caller sets the `current_on_token` ContextVar (e.g. the
Streamlit UI), invoke_structured switches from the buffered
`with_structured_output` path to a raw-JSON streaming path that emits
chunks via the callback and parses the buffer at the end. Node code is
untouched.
"""
from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar
from typing import Callable, Optional, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    DEFAULT_TEMPERATURE,
    GEMINI_MODEL,
    GROQ_MODEL,
    MAX_RETRIES,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

T = TypeVar("T", bound=BaseModel)


# ContextVar set by the streaming layer. When None (default), invoke_structured
# uses the original buffered path.
current_on_token: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
    "current_on_token", default=None
)


def get_llm(temperature: float = DEFAULT_TEMPERATURE) -> BaseChatModel:
    """Build a chat model from the live LLM_PROVIDER env var.

    Env vars are read live (not cached at import) so the Streamlit radio can
    switch providers mid-session by writing os.environ before invoking.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_key:
            raise RuntimeError(
                "GEMINI_API_KEY missing. Set it in .env "
                "(free key: https://aistudio.google.com/apikey)"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=temperature,
            google_api_key=gemini_key,
        )

    if provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            raise RuntimeError(
                "GROQ_API_KEY missing. Set it in .env "
                "(free key: https://console.groq.com/keys)"
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=temperature,
            groq_api_key=groq_key,
        )

    if provider == "openrouter":
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if not or_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY missing. Set it in .env "
                "(free key: https://openrouter.ai/keys). "
                "Pick a :free model in OPENROUTER_MODEL — see "
                "https://openrouter.ai/models?max_price=0"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL),
            temperature=temperature,
            api_key=or_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            default_headers={
                "HTTP-Referer": "https://github.com/cantordust-task1",
                "X-Title": "Nepal Import Compliance Drafter",
            },
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Use 'gemini', 'groq', or 'openrouter'."
    )


def get_active_provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").lower()


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s).strip()


def _extract_json_blob(s: str) -> str:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in streamed buffer")
    return s[start : end + 1]


def _extract_delta(chunk) -> str:
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
    from src.prompts import JSON_ONLY_GUARD

    sys2 = system_prompt + JSON_ONLY_GUARD.format(
        json_schema=json.dumps(schema.model_json_schema(), separators=(",", ":"))
    )
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
    """Return a validated instance of `schema`. Streams via on_token (or the
    current_on_token ContextVar) when set, else uses with_structured_output."""
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
    llm = get_llm(temperature=temperature)
    result = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return result.content if hasattr(result, "content") else str(result)
