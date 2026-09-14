# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import verify_native_planner_e2e

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_missing_transitions_fixture(root: Path) -> tuple[Path, Path]:
    run_dir = root / "run"
    evidence_dir = root / "evidence"
    metrics_dir = run_dir / "metrics" / "async"
    metrics_dir.mkdir(parents=True)
    evidence_dir.mkdir()

    _write_json(
        run_dir / "terminal-batch.json",
        {
            "id": "batch-test",
            "status": "completed",
            "request_counts": {"completed": 100, "failed": 0, "total": 100},
        },
    )
    _write_json(
        run_dir / "result-validation.json",
        {
            "valid": True,
            "downloaded_output_lines": 100,
            "unique_custom_ids": 100,
        },
    )
    (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (metrics_dir / "sample.prom").write_text(
        "llm_d_async_async_dispatched_requests_total"
        '{pool_name="dynamo-batch"} 403\n',
        encoding="utf-8",
    )
    _write_json(
        metrics_dir / "sample.json",
        {"observed_at": "2026-08-28T21:38:17.330Z"},
    )

    state = {
        "observed_at": "2026-08-28T21:35:35Z",
        "adapter_spec": 0,
        "ready_worker_pods": 0,
        "worker_ready_replicas": 0,
        "dgd_ready": "False",
        "lease_csv": '"llm-d.ai/v1alpha1","dynamo-batch","0","1","decision"',
    }
    (evidence_dir / "state.jsonl").write_text(
        json.dumps(state) + "\n", encoding="utf-8"
    )
    _write_json(
        evidence_dir / "dgdsa.before.json",
        {
            "metadata": {"generation": 1, "resourceVersion": "1"},
            "spec": {"replicas": 0},
            "status": {"replicas": 0},
        },
    )
    _write_json(
        evidence_dir / "dgdsa.after.json",
        {
            "metadata": {"generation": 1, "resourceVersion": "1"},
            "spec": {"replicas": 0},
            "status": {"replicas": 0},
        },
    )
    (evidence_dir / "dgdsa.watch.jsonstream").write_text(
        json.dumps({"object": {"spec": {"replicas": 0}}}) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "planner.log").write_text("", encoding="utf-8")
    (evidence_dir / "redis.after.txt").write_text(
        "llm-d.ai/v1alpha1\ndynamo-batch\n0\n1\ndecision\n",
        encoding="utf-8",
    )
    (evidence_dir / "redis.after.pttl-ms.txt").write_text("1000\n", encoding="utf-8")
    (evidence_dir / "async-metrics.before.txt").write_text(
        "llm_d_async_async_dispatched_requests_total 400\n"
        "llm_d_async_async_successful_requests_total 400\n",
        encoding="utf-8",
    )
    (evidence_dir / "async-metrics.after.txt").write_text(
        "llm_d_async_async_dispatched_requests_total 500\n"
        "llm_d_async_async_successful_requests_total 500\n"
        "llm_d_async_async_broker_backlog 0\n"
        "llm_d_async_async_inflight_requests 0\n"
        "llm_d_async_async_queue_depth 0\n"
        "llm_d_async_async_drain_limit_rps 0\n",
        encoding="utf-8",
    )
    return run_dir, evidence_dir


def test_missing_transitions_produce_a_normal_failure_report(tmp_path: Path) -> None:
    run_dir, evidence_dir = _write_missing_transitions_fixture(tmp_path)

    result = verify_native_planner_e2e.verify(run_dir, evidence_dir)

    assert result["all_passed"] is False
    assert result["assertions"]["scale_preceded_worker_readiness"] is False
    assert result["assertions"]["lease_remained_closed_until_ready"] is False
    assert result["assertions"]["dispatch_started_after_positive_lease"] is False
    assert result["assertions"]["terminal_zero_observed_after_positive"] is False
    assert result["timeline"]["adapter_one"] is None
    assert result["timeline"]["worker_ready"] is None
    assert result["timeline"]["positive_lease"] is None
    assert result["timeline"]["terminal_zero_lease"] is None
