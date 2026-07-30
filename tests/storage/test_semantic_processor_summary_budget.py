# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.storage.queuefs import semantic_processor as semantic_processor_module
from openviking.storage.queuefs.semantic_dag import SummaryBudget
from openviking.storage.queuefs.semantic_processor import SemanticProcessor


class RecordingVLM:
    """VLM mock that records every prompt sent to get_completion_async."""

    def __init__(self, response: str = "summary response"):
        self.prompts = []
        self._response = response

    def is_available(self):
        return True

    async def get_completion_async(self, prompt):
        self.prompts.append(prompt)
        return self._response


def _make_config(vlm, max_summary_input_chars: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        vlm=vlm,
        semantic=SimpleNamespace(
            max_file_content_chars=1000,
            max_skeleton_chars=500,
            max_overview_prompt_chars=10_000,
            overview_batch_size=10,
            max_summary_input_chars=max_summary_input_chars,
            abstract_max_chars=256,
            overview_max_chars=4000,
            memory_chunk_chars=2000,
            memory_chunk_overlap=200,
        ),
        code=SimpleNamespace(
            code_summary_mode="llm",
        ),
        output_language_override="en",
    )


def _patch_config(monkeypatch, config):
    monkeypatch.setattr(
        semantic_processor_module,
        "get_openviking_config",
        lambda: config,
    )


# ── SummaryBudget unit tests ─────────────────────────────────────────


def test_summary_budget_unlimited():
    budget = SummaryBudget(0)
    assert budget.unlimited is True
    assert budget.exhausted is False
    assert budget.try_reserve(999_999) is True


def test_summary_budget_atomic_reserve_and_exact_consumption():
    """try_reserve is atomic: exact debit, exhaustion at 0, oversize denied."""
    budget = SummaryBudget(100)
    assert budget.try_reserve(30) is True
    assert budget.remaining == 70
    assert budget.try_reserve(70) is True
    assert budget.remaining == 0
    assert budget.exhausted is True
    # Any further request of any positive size fails
    assert budget.try_reserve(1) is False
    # Zero cost always passes (no-op)
    assert budget.try_reserve(0) is True


def test_summary_budget_oversize_call_denied_without_consuming():
    """A single call larger than remaining budget is denied; budget intact."""
    budget = SummaryBudget(50)
    assert budget.try_reserve(100) is False
    assert budget.remaining == 50
    assert budget.exhausted is False
    # A smaller call can still succeed
    assert budget.try_reserve(50) is True
    assert budget.exhausted is True


def test_summary_budget_concurrent_no_overspend():
    """Concurrent try_reserve from many threads never exceeds budget."""
    budget = SummaryBudget(1000)
    success_count = 0
    lock = threading.Lock()

    def worker(n: int, size: int) -> None:
        nonlocal success_count
        local_success = 0
        for _ in range(n):
            if budget.try_reserve(size):
                local_success += 1
        with lock:
            success_count += local_success

    threads = [threading.Thread(target=worker, args=(200, 3)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_spent = success_count * 3
    assert total_spent <= 1000
    assert budget.remaining >= 0
    assert budget.remaining < 3  # next call would fail (or already did)


def test_summary_budget_first_rejection_warning_fires_once():
    """take_first_rejection_warning returns True exactly once, monotonically.

    Consecutive rejections must not re-trigger the warning, and concurrent
    callers must not both get True.
    """
    budget = SummaryBudget(10)
    # No rejection yet → False
    assert budget.take_first_rejection_warning() is False
    # First rejection sets the flag
    assert budget.try_reserve(100) is False
    # First take returns True
    assert budget.take_first_rejection_warning() is True
    # Subsequent takes return False (already consumed)
    assert budget.take_first_rejection_warning() is False
    # More rejections do not re-arm the warning
    assert budget.try_reserve(100) is False
    assert budget.try_reserve(100) is False
    assert budget.take_first_rejection_warning() is False


def test_summary_budget_concurrent_first_rejection_warning_once():
    """Concurrent take_first_rejection_warning calls — exactly one wins."""
    budget = SummaryBudget(5)
    budget.try_reserve(100)  # trigger rejection

    warning_count = 0
    lock = threading.Lock()

    def worker():
        nonlocal warning_count
        if budget.take_first_rejection_warning():
            with lock:
                warning_count += 1

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert warning_count == 1, f"expected 1 warning, got {warning_count}"


def test_summary_budget_separate_instances_are_isolated():
    """Two SummaryBudget instances do not share state (task isolation)."""
    budget_a = SummaryBudget(100)
    budget_b = SummaryBudget(100)
    budget_a.try_reserve(80)
    assert budget_a.remaining == 20
    assert budget_b.remaining == 100


# ── SemanticProcessor integration tests ──────────────────────────────


PROMPT_LEN = 40  # fixed prompt length used in tests with patched render_prompt


@pytest.mark.asyncio
async def test_text_summary_budget_exact_accounting_and_skipping(monkeypatch, caplog):
    """File summaries charge exact rendered-prompt length; over-budget calls skip VLM.

    Proves:
    - Cumulative VLM input chars == budget exactly (not less, not more)
    - Files that exceed remaining budget get empty summary but full content
      (vectorization still works)
    - Exactly one warning per task (not per rejected call), with limit/requested/remaining
    """
    import logging

    vlm = RecordingVLM()
    config = _make_config(vlm)
    _patch_config(monkeypatch, config)

    # Fixed-length prompts for deterministic budget math
    monkeypatch.setattr(
        semantic_processor_module,
        "render_prompt",
        lambda name, vals: "x" * PROMPT_LEN,
    )

    content = "hello world"
    mock_fs = MagicMock()
    mock_fs.read_file = AsyncMock(return_value=content)
    monkeypatch.setattr(
        semantic_processor_module,
        "get_viking_fs",
        lambda: mock_fs,
    )

    processor = SemanticProcessor()
    BUDGET = PROMPT_LEN * 3  # exactly 3 calls fit
    budget = SummaryBudget(BUDGET)

    with caplog.at_level(logging.WARNING, logger="openviking"):
        results = []
        for i in range(5):
            result = await processor._generate_text_summary(
                f"viking://resources/test/file{i}.txt",
                f"file{i}.txt",
                asyncio.Semaphore(1),
                summary_budget=budget,
            )
            results.append(result)

    # Exactly 3 calls fit in the budget
    assert len(vlm.prompts) == 3
    # Cumulative input matches budget exactly
    assert sum(len(p) for p in vlm.prompts) == BUDGET
    # 3 non-empty summaries, 2 empty
    assert sum(1 for r in results if r["summary"]) == 3
    assert sum(1 for r in results if not r["summary"]) == 2
    # All files retain full content for vectorization
    assert all(r["content"] == content for r in results)
    # Budget is exactly exhausted
    assert budget.remaining == 0
    assert budget.exhausted is True
    # Exactly one warning for 2 consecutive rejections, with numeric details
    warning_records = [r for r in caplog.records if "VLM summary input limit" in r.message]
    assert len(warning_records) == 1, (
        f"Expected exactly 1 budget warning, got {len(warning_records)}: "
        f"{[r.message for r in warning_records]}"
    )
    assert str(BUDGET) in warning_records[0].message  # limit
    assert str(PROMPT_LEN) in warning_records[0].message  # requested


@pytest.mark.asyncio
async def test_text_summary_ast_only_does_not_consume_budget(monkeypatch):
    """AST-only code mode never calls VLM, so budget is not consumed."""
    vlm = RecordingVLM()
    config = _make_config(vlm)
    config.code.code_summary_mode = "ast"
    _patch_config(monkeypatch, config)

    # 200-line file triggers AST path
    content = "\n".join([f"line {i}" for i in range(200)])
    mock_fs = MagicMock()
    mock_fs.read_file = AsyncMock(return_value=content)
    monkeypatch.setattr(
        semantic_processor_module,
        "get_viking_fs",
        lambda: mock_fs,
    )

    processor = SemanticProcessor()
    budget = SummaryBudget(10_000)

    result = await processor._generate_text_summary(
        "viking://resources/test/main.py",
        "main.py",
        asyncio.Semaphore(1),
        summary_budget=budget,
    )

    assert len(vlm.prompts) == 0
    assert budget.remaining == 10_000  # nothing charged
    assert result["summary"]  # AST skeleton is non-empty


@pytest.mark.asyncio
async def test_overview_and_file_summaries_share_one_budget(monkeypatch):
    """File summaries and directory overview draw from the same budget."""
    vlm = RecordingVLM(response="ok")
    config = _make_config(vlm)
    _patch_config(monkeypatch, config)

    # Fixed prompt length for both summaries and overviews
    monkeypatch.setattr(
        semantic_processor_module,
        "render_prompt",
        lambda name, vals: "x" * PROMPT_LEN,
    )

    content = "x" * 10
    mock_fs = MagicMock()
    mock_fs.read_file = AsyncMock(return_value=content)
    monkeypatch.setattr(
        semantic_processor_module,
        "get_viking_fs",
        lambda: mock_fs,
    )

    processor = SemanticProcessor()
    # Budget fits exactly 4 calls total. 2 file summaries + 1 overview = 3,
    # leaving room. If overview came from a separate budget, we'd see more calls.
    BUDGET = PROMPT_LEN * 4
    budget = SummaryBudget(BUDGET)

    # Generate 2 file summaries (2 calls)
    for i in range(2):
        await processor._generate_text_summary(
            f"viking://resources/test/file{i}.txt",
            f"file{i}.txt",
            asyncio.Semaphore(1),
            summary_budget=budget,
        )

    file_calls = len(vlm.prompts)
    assert file_calls == 2

    # Generate overview for a directory (1 call — fits in remaining budget)
    file_summaries = [{"name": f"f{i}.txt", "summary": f"summary{i}"} for i in range(3)]
    await processor._generate_overview(
        "viking://resources/test",
        file_summaries=file_summaries,
        children_abstracts=[],
        summary_budget=budget,
    )

    total_calls = len(vlm.prompts)
    assert total_calls == 3  # 2 file + 1 overview
    total_input = sum(len(p) for p in vlm.prompts)
    assert total_input == PROMPT_LEN * 3
    assert total_input <= BUDGET


@pytest.mark.asyncio
async def test_batched_overview_total_input_never_exceeds_budget(monkeypatch):
    """Batched overview: sum of all batch + merge prompt chars <= budget."""
    vlm = RecordingVLM(response="partial")
    config = _make_config(vlm)
    config.semantic.max_overview_prompt_chars = 200  # force batching
    config.semantic.overview_batch_size = 2
    _patch_config(monkeypatch, config)

    # Prompts proportional to input size (not fixed — so batches vary)
    def render(name, vals):
        return f"[{name}] {vals['file_summaries']}"

    monkeypatch.setattr(semantic_processor_module, "render_prompt", render)

    processor = SemanticProcessor()
    file_summaries = [{"name": f"f{i}.txt", "summary": "s" * 20} for i in range(6)]

    BUDGET = 300
    budget = SummaryBudget(BUDGET)

    await processor._generate_overview(
        "viking://resources/test",
        file_summaries=file_summaries,
        children_abstracts=[],
        summary_budget=budget,
    )

    # Core invariant: cumulative rendered prompt chars never exceed budget
    total_input = sum(len(p) for p in vlm.prompts)
    assert total_input <= BUDGET, f"Cumulative VLM input ({total_input}) exceeded budget ({BUDGET})"
