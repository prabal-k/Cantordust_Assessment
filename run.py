"""CLI entrypoint.

Usage:
    python run.py --pdf1 <path> --pdf2 <path> --nepqa <path> [--out outputs/]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import LLM_PROVIDER
from src.graph import build_graph
from src.state import AgentState

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cantordust Task 1 — Nepal compliance drafter."
    )
    p.add_argument("--pdf1", required=True, help="Path to first manufacturer PDF")
    p.add_argument("--pdf2", required=True, help="Path to second manufacturer PDF")
    p.add_argument("--nepqa", required=True, help="Path to NEPQA 2025 PDF")
    p.add_argument(
        "--out", default="outputs", help="Output directory (default: outputs/)"
    )
    p.add_argument(
        "--retries",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Critic self-correction retries (1-3, default 2). "
        "If the critic flags issues, the agent patches the flagged fields and "
        "re-drafts up to this many times.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for label, path in [("PDF1", args.pdf1), ("PDF2", args.pdf2), ("NEPQA", args.nepqa)]:
        if not Path(path).exists():
            console.print(f"[red]ERROR: {label} not found at {path}[/red]")
            return 2

    initial: AgentState = {
        "pdf1_path": args.pdf1,
        "pdf2_path": args.pdf2,
        "nepqa_path": args.nepqa,
        "pdf1_name": Path(args.pdf1).stem,
        "pdf2_name": Path(args.pdf2).stem,
        "nepqa_name": Path(args.nepqa).stem,
        "interface": "cli",
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "max_retries": args.retries,
        "retry_count": 0,
        "patch_history": [],
        "best_attempt": {},
    }

    console.print(
        Panel.fit(
            "[bold cyan]Cantordust Task 1 — Nepal Compliance Drafter[/bold cyan]\n"
            "10-node LangGraph pipeline starting...",
            border_style="cyan",
        )
    )

    graph = build_graph(use_checkpointer=False)
    run_config = {
        "run_name": f"cantordust-task1-cli-{initial['run_timestamp']}",
        "tags": ["cantordust-task1", "cli", f"provider={LLM_PROVIDER}"],
        "metadata": {
            "interface": "cli",
            "provider": LLM_PROVIDER,
            "run_timestamp": initial["run_timestamp"],
        },
    }
    result: AgentState = graph.invoke(initial, config=run_config)

    # Report
    table = Table(title="Pipeline result", show_lines=False)
    table.add_column("Artifact", style="cyan")
    table.add_column("Value")
    table.add_row("Variant relationship", result.get("variant_decision").relationship.value if result.get("variant_decision") else "—")
    table.add_row(
        "Chosen family",
        result["chosen_record"].family_label if result.get("chosen_record") else "—",
    )
    table.add_row("Mismatches", str(len(result.get("mismatches", []))))
    table.add_row("Coverage items", str(len(result.get("coverage", []))))
    table.add_row("Critic flags", str(len(result.get("critic_flags", []))))
    table.add_row(
        "Critic retries",
        f"{result.get('retry_count', 0)} / {result.get('max_retries', args.retries)}",
    )
    table.add_row("Markdown draft", result.get("draft_md_path", "—"))
    table.add_row(
        "PDF draft",
        result.get("draft_pdf_path") or "skipped (WeasyPrint unavailable)",
    )
    console.print(table)

    if result.get("ask_factory_list"):
        console.print("\n[bold yellow]Ask the factory for:[/bold yellow]")
        for item in result["ask_factory_list"]:
            console.print(f"  • {item}")

    console.print("\n[green]Done.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
