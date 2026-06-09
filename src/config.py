"""Centralized config. Loads env vars, defines useful-page slices, output paths."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache"
DEBUG_DIR = OUTPUTS_DIR / "debug"

for d in (OUTPUTS_DIR, CACHE_DIR, DEBUG_DIR):
    d.mkdir(parents=True, exist_ok=True)
# Avoid sending all 107 pages to the LLM. Verified by manual inspection.
PDF1_USEFUL_PAGES = list(range(1, 9))    # cover + factory declaration + product info + model list
PDF2_USEFUL_PAGES = list(range(1, 5))    # full 4-page certificate
NEPQA_USEFUL_PAGES = [18, 19]            # Section 1.4 PV Inverter / Grid Connected Inverter
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# OpenRouter — OpenAI-compatible gateway. Default = a free-tier-eligible model.
# See https://openrouter.ai/models?max_price=0 for current free list.
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "qwen/qwen3-next-80b-a3b-instruct:free"
)
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_TEMPERATURE = 0.1
CRITIC_TEMPERATURE = 0.0
MAX_RETRIES = 3
VARIANT_AGENT_MAX_TOOL_CALLS = int(os.getenv("VARIANT_AGENT_MAX_TOOL_CALLS", "8"))
# LangChain auto-traces if LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY are set.
# We just surface the booleans for app.py to show status.
LANGSMITH_TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "cantordust-nepal-compliance")
# Ensure the project name reaches LangChain even if user only set TRACING_V2:
if LANGSMITH_TRACING_ENABLED and "LANGCHAIN_PROJECT" not in os.environ:
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
