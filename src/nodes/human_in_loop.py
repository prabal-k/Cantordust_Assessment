"""Node 6: human-in-the-loop.

CLI: blocking `input()`.
Streamlit: raises `NeedsHumanInput`; the UI catches it, renders a radio button,
then resumes the graph from the LangGraph checkpoint.
"""
from __future__ import annotations

from src.schemas import HumanChoice
from src.state import AgentState


class NeedsHumanInput(Exception):
    """Streamlit catches this to pause the graph for user input."""

    def __init__(self, p1_label: str, p2_label: str):
        super().__init__("Awaiting human family choice")
        self.p1_label = p1_label
        self.p2_label = p2_label


def _label(record) -> str:
    if not record.model_numbers:
        return record.family_label
    sample = record.model_numbers[0].value
    return f"{record.family_label} ({sample})"


def human_in_loop_node(state: AgentState) -> AgentState:
    p1 = state["pdf1_record"]
    p2 = state["pdf2_record"]
    interface = state.get("interface", "cli")

    # Pre-supplied choice (e.g. Streamlit resume) wins.
    pre = state.get("human_choice")
    if pre is not None:
        chosen = p1 if pre.chosen_family == "pdf1" else p2
        return {"chosen_record": chosen}

    if interface == "streamlit":
        raise NeedsHumanInput(p1_label=_label(p1), p2_label=_label(p2))

    # CLI flow
    p1_label = _label(p1)
    p2_label = _label(p2)
    print("\n" + "=" * 60)
    print("  Variant detector flagged the two PDFs as different products.")
    print("  Which product family is this shipment about?")
    print("=" * 60)
    print(f"  [1] {p1_label}   (from PDF1)")
    print(f"  [2] {p2_label}   (from PDF2)")
    while True:
        answer = input("Enter 1 or 2: ").strip()
        if answer in ("1", "2"):
            break
        print("  Please type 1 or 2.")
    chosen_family = "pdf1" if answer == "1" else "pdf2"
    chosen = p1 if chosen_family == "pdf1" else p2
    return {
        "human_choice": HumanChoice(chosen_family=chosen_family, rationale="CLI input"),
        "chosen_record": chosen,
    }


def auto_choose_node(state: AgentState) -> AgentState:
    """Used when variant decision is SAME_PRODUCT or VARIANT — no human needed.

    Default to PDF2 (typically the certificate of conformity, richer for compliance)
    if it exists; otherwise PDF1.
    """
    p1 = state["pdf1_record"]
    p2 = state.get("pdf2_record")
    chosen = p2 if p2 is not None else p1
    chosen_family = "pdf2" if chosen is p2 else "pdf1"
    return {
        "human_choice": HumanChoice(
            chosen_family=chosen_family, rationale="auto: variant_decision did not require human"
        ),
        "chosen_record": chosen,
    }
