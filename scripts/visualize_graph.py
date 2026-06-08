"""Render the LangGraph StateGraph as a Mermaid PNG + Mermaid source.

Outputs:
  - docs/graph.png         Mermaid-rendered PNG (via langgraph's mermaid-ink call)
  - docs/graph.mmd         Mermaid source for embedding in markdown
  - docs/graph_ascii.txt   ASCII fallback diagram
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Avoid hitting LLM env-var guards just to build the graph.
os.environ.setdefault("GEMINI_API_KEY", "stub-for-graph-render")
os.environ.setdefault("GROQ_API_KEY", "stub-for-graph-render")

from src.graph import build_graph  # noqa: E402


def main() -> int:
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph(use_checkpointer=False)
    g = graph.get_graph()

    # 1. Mermaid source
    mmd_path = docs_dir / "graph.mmd"
    mermaid_src = g.draw_mermaid()
    mmd_path.write_text(mermaid_src, encoding="utf-8")
    print(f"Wrote Mermaid source -> {mmd_path}")

    # 2. PNG via mermaid.ink HTTP rendering (LangGraph's built-in helper)
    png_path = docs_dir / "graph.png"
    try:
        png_bytes = g.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"Wrote PNG -> {png_path}")
    except Exception as e:
        print(f"PNG render failed (network needed for mermaid.ink): {e}")

    # 3. ASCII fallback
    ascii_path = docs_dir / "graph_ascii.txt"
    try:
        ascii_text = g.draw_ascii()
        ascii_path.write_text(ascii_text, encoding="utf-8")
        print(f"Wrote ASCII -> {ascii_path}")
    except Exception as e:
        print(f"ASCII render skipped: {e}")

    print("\nMermaid source (preview):\n")
    print(mermaid_src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
