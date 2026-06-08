"""Unit test for patch_extractor: mocks the LLM, verifies state updates."""
from __future__ import annotations

from src.schemas import CriticFlag, FieldClaim


def test_patch_extractor_increments_retry_and_clears_downstream(
    monkeypatch, sample_record_pdf1
):
    from src.nodes import patch_extractor

    flag = CriticFlag(
        section="electrical.ac_voltage_v",
        claim_excerpt="230",
        issue="Voltage not supported by source page",
        suggested_action="Re-read page 7",
    )
    patched_record = sample_record_pdf1.model_copy()

    def fake_invoke_structured(schema, system_prompt, user_prompt, **kwargs):
        # Simulate LLM returning a valid ProductRecord
        return patched_record

    monkeypatch.setattr(
        patch_extractor, "invoke_structured", fake_invoke_structured
    )

    initial_state = {
        "chosen_record": sample_record_pdf1,
        "critic_flags": [flag],
        "ask_factory_list": ["some ask"],
        "draft_markdown": "previous draft",
        "retry_count": 0,
        "patch_history": [],
        "pdf1_pages": {1: "stub", 2: "stub", 3: "stub", 4: "stub",
                       5: "stub", 6: "stub", 7: "stub", 8: "stub"},
        "pdf2_pages": {1: "stub", 2: "stub", 3: "stub", 4: "stub"},
    }

    update = patch_extractor.patch_extractor_node(initial_state)

    assert update["retry_count"] == 1
    assert update["critic_flags"] == []
    assert update["ask_factory_list"] == []
    assert update["draft_markdown"] == ""
    assert update["chosen_record"].source_doc == sample_record_pdf1.source_doc
    assert update["chosen_record"].family_label == sample_record_pdf1.family_label
    assert len(update["patch_history"]) == 1
    assert update["patch_history"][0]["attempt"] == 1
    assert update["patch_history"][0]["flag_count_before"] == 1


def test_patch_extractor_noop_when_no_flags(sample_record_pdf1):
    from src.nodes.patch_extractor import patch_extractor_node

    update = patch_extractor_node(
        {
            "chosen_record": sample_record_pdf1,
            "critic_flags": [],
            "retry_count": 0,
        }
    )
    # No flags → only retry_count advances
    assert update == {"retry_count": 1}
