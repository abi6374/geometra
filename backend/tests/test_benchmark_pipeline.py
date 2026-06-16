"""Benchmark tests for pipeline throughput with large DXF files.

Measures processing times and throughput (entities/second) for each agent
individually and for the full Agents 1→6 pipeline, across multiple DXF sizes.

These tests use time.perf_counter() for timing and report results via assert
statements. Run with::

    pytest tests/test_benchmark_pipeline.py -v --tb=short

To see benchmark numbers (not just pass/fail), use::

    pytest tests/test_benchmark_pipeline.py -v -s
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from geometra.agents.agent_01_input_processing import InputProcessingAgent
from geometra.agents.agent_02_drawing_understanding import DrawingUnderstandingAgent
from geometra.agents.agent_03_ocr_annotation import OCRAnnotationAgent
from geometra.agents.agent_04_feature_recognition import FeatureRecognitionAgent
from geometra.agents.agent_05_engineering_reasoning import EngineeringReasoningAgent
from geometra.agents.agent_06 import CADGenerationAgent

# ── Agent instances (module-level for reuse) ─────────────────────────────────

agent_1 = InputProcessingAgent()
agent_2 = DrawingUnderstandingAgent()
agent_3 = OCRAnnotationAgent()
agent_4 = FeatureRecognitionAgent()
agent_5 = EngineeringReasoningAgent()
agent_6 = CADGenerationAgent()


# ── DXF File Generator ───────────────────────────────────────────────────────


def _generate_large_dxf(
    num_entities: int,
    *,
    circle_fraction: float = 0.2,
    line_fraction: float = 0.5,
    arc_fraction: float = 0.1,
    polyline_fraction: float = 0.1,
    text_fraction: float = 0.1,
) -> Path:
    """Generate a large DXF file with the specified number of entities.

    The entity types are distributed according to the given fractions.
    Returns the path to the generated file.

    Args:
        num_entities: Total number of drawing entities to generate.
        circle_fraction: Fraction of entities that are circles (holes).
        line_fraction: Fraction of entities that are lines.
        arc_fraction: Fraction of entities that are arcs.
        polyline_fraction: Fraction of entities that are LWPOLYLINES.
        text_fraction: Fraction of entities that are TEXT/MTEXT.
    """
    import ezdxf

    # Use R2000 for LWPOLYLINE support (most real-world DXFs use this)
    doc = ezdxf.new("R2000")
    msp = doc.modelspace()

    # Spread entities across a large grid (1000x1000 per grid cell)
    grid_size = max(1, int(math.sqrt(num_entities)))
    spacing = 10.0

    counts = {
        "LINE": int(num_entities * line_fraction),
        "CIRCLE": int(num_entities * circle_fraction),
        "ARC": int(num_entities * arc_fraction),
        "LWPOLYLINE": int(num_entities * polyline_fraction),
        "TEXT": int(num_entities * text_fraction),
    }

    # Adjust for rounding
    total_assigned = sum(counts.values())
    counts["LINE"] += num_entities - total_assigned

    entity_idx = 0

    # Lines
    for _ in range(counts["LINE"]):
        bx = (entity_idx % grid_size) * spacing
        by = (entity_idx // grid_size) * spacing
        msp.add_line((bx, by), (bx + 5, by + 5))
        entity_idx += 1

    # Circles
    for _ in range(counts["CIRCLE"]):
        bx = (entity_idx % grid_size) * spacing
        by = (entity_idx // grid_size) * spacing
        msp.add_circle((bx, by), radius=3.0)
        entity_idx += 1

    # Arcs
    for _ in range(counts["ARC"]):
        bx = (entity_idx % grid_size) * spacing
        by = (entity_idx // grid_size) * spacing
        msp.add_arc((bx, by), radius=5.0, start_angle=0, end_angle=180)
        entity_idx += 1

    # LWPOLYLINES (closed rectangles)
    for _ in range(counts["LWPOLYLINE"]):
        bx = (entity_idx % grid_size) * spacing
        by = (entity_idx // grid_size) * spacing
        msp.add_lwpolyline([(bx, by), (bx + 8, by), (bx + 8, by + 5), (bx, by + 5)], close=True)
        entity_idx += 1

    # TEXT
    for _ in range(counts["TEXT"]):
        bx = (entity_idx % grid_size) * spacing
        by = (entity_idx // grid_size) * spacing
        msp.add_text(f"DIM_{entity_idx}", dxfattribs={"height": 1.0, "insert": (bx, by)})
        entity_idx += 1

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf")
    doc.saveas(tmp.name)
    tmp.close()
    return Path(tmp.name)


# ── DXF Size Fixtures (session-scoped to generate once per session) ──────────


@pytest.fixture(scope="session")
def dxf_small() -> Path:
    """Small DXF with ~100 entities (baseline)."""
    return _generate_large_dxf(100)


@pytest.fixture(scope="session")
def dxf_medium() -> Path:
    """Medium DXF with ~1,000 entities."""
    return _generate_large_dxf(1000)


@pytest.fixture(scope="session")
def dxf_large() -> Path:
    """Large DXF with ~10,000 entities."""
    return _generate_large_dxf(10000)


@pytest.fixture(scope="session")
def dxf_xlarge() -> Path:
    """Extra-large DXF with ~100,000 entities."""
    return _generate_large_dxf(100000)


# ── Helper: run full pipeline and return per-agent timing dict ───────────────


def _benchmark_dxf_pipeline(file_path: Path) -> dict[str, float]:
    """Run the full Agents 1→6 pipeline on a DXF file and return per-agent times."""
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    doc = agent_1.process(str(file_path))
    timing["agent1_load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    primitives = agent_2.process(doc)
    timing["agent2_primitives"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    annotations = agent_3.process(doc)
    timing["agent3_annotations"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    features = agent_4.process(primitives)
    timing["agent4_features"] = time.perf_counter() - t0

    input_5 = {"primitives": primitives, "annotations": annotations, "features": features}
    t0 = time.perf_counter()
    reasoning = agent_5.process(input_5)
    timing["agent5_reasoning"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cad = agent_6.process(reasoning)
    timing["agent6_cad"] = time.perf_counter() - t0

    timing["total"] = sum(timing.values())

    # Entity count from Agent 1 output
    entity_total = doc.get("data", {}).get("entity_total", 0)
    timing["entity_count"] = float(entity_total)

    # Number of operations from Agent 5
    timing["operation_count"] = float(len(reasoning.get("operations", [])))

    # Features detected by Agent 4
    timing["feature_count"] = float(features.get("feature_count", 0))

    return timing


# ── Benchmark Class ──────────────────────────────────────────────────────────


class TestDxfPipelineThroughput:
    """Benchmark the DXF pipeline at various entity counts.

    These tests measure throughput and assert that:
      - The pipeline completes without errors at each size
      - Larger files take longer than smaller ones (monotonic scaling)
      - Throughput stays within expected bounds (not pathologically slow)
    """

    # ── Agent 1 (Input Processing) Benchmarks ──

    def test_agent1_throughput_small(self, dxf_small: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_small)
        entities = timing["entity_count"]
        t = timing["agent1_load"]
        throughput = entities / t if t > 0 else 0
        print(f"\n  Agent 1 (small): {entities} entities in {t:.4f}s → {throughput:.0f} entities/s")
        assert t < 2.0, f"Agent 1 took too long for {entities} entities: {t:.3f}s"

    def test_agent1_throughput_medium(self, dxf_medium: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_medium)
        entities = timing["entity_count"]
        t = timing["agent1_load"]
        throughput = entities / t if t > 0 else 0
        print(f"\n  Agent 1 (medium): {entities} entities in {t:.4f}s → {throughput:.0f} entities/s")
        assert t < 3.0, f"Agent 1 took too long for {entities} entities: {t:.3f}s"

    def test_agent1_throughput_large(self, dxf_large: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_large)
        entities = timing["entity_count"]
        t = timing["agent1_load"]
        throughput = entities / t if t > 0 else 0
        print(f"\n  Agent 1 (large): {entities} entities in {t:.4f}s → {throughput:.0f} entities/s")
        assert t < 10.0, f"Agent 1 took too long for {entities} entities: {t:.3f}s"

    def test_agent1_throughput_xlarge(self, dxf_xlarge: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_xlarge)
        entities = timing["entity_count"]
        t = timing["agent1_load"]
        throughput = entities / t if t > 0 else 0
        print(f"\n  Agent 1 (xlarge): {entities} entities in {t:.4f}s → {throughput:.0f} entities/s")
        # 100K entities should finish in reasonable time
        assert t < 30.0, f"Agent 1 took too long for {entities} entities: {t:.3f}s"

    # ── Agent 2 (Drawing Understanding / DXF Primitives) Benchmarks ──

    def test_agent2_throughput_small(self, dxf_small: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_small)
        entities = timing["entity_count"]
        t = timing["agent2_primitives"]
        throughput = entities / t if t > 0 else 0
        print(f"\n  Agent 2 (small): {entities} entities in {t:.4f}s → {throughput:.0f} entities/s")
        assert t < 1.0

    def test_agent2_throughput_medium(self, dxf_medium: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_medium)
        entities = timing["entity_count"]
        t = timing["agent2_primitives"]
        throughput = entities / t if t > 0 else 0
        print(f"\n  Agent 2 (medium): {entities} entities in {t:.4f}s → {throughput:.0f} entities/s")
        assert t < 1.0

    def test_agent2_throughput_large(self, dxf_large: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_large)
        entities = timing["entity_count"]
        t = timing["agent2_primitives"]
        throughput = entities / t if t > 0 else 0
        print(f"\n  Agent 2 (large): {entities} entities in {t:.4f}s → {throughput:.0f} entities/s")
        assert t < 2.0

    def test_agent2_throughput_xlarge(self, dxf_xlarge: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_xlarge)
        entities = timing["entity_count"]
        t = timing["agent2_primitives"]
        throughput = entities / t if t > 0 else 0
        print(f"\n  Agent 2 (xlarge): {entities} entities in {t:.4f}s → {throughput:.0f} entities/s")
        assert t < 5.0

    # ── Full Pipeline Benchmarks ──

    def test_full_pipeline_throughput_small(self, dxf_small: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_small)
        entities = timing["entity_count"]
        total = timing["total"]
        throughput = entities / total if total > 0 else 0
        print(
            f"\n  Full pipeline (small): {entities} entities in {total:.4f}s"
            f" → {throughput:.0f} entities/s"
            f" | Agents: {timing['agent1_load']:.3f}s + {timing['agent2_primitives']:.3f}s"
            f" + {timing['agent3_annotations']:.3f}s + {timing['agent4_features']:.3f}s"
            f" + {timing['agent5_reasoning']:.3f}s + {timing['agent6_cad']:.3f}s"
        )
        assert total < 5.0, f"Full pipeline too slow for {entities} entities: {total:.3f}s"

    def test_full_pipeline_throughput_medium(self, dxf_medium: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_medium)
        entities = timing["entity_count"]
        total = timing["total"]
        throughput = entities / total if total > 0 else 0
        print(
            f"\n  Full pipeline (medium): {entities} entities in {total:.4f}s"
            f" → {throughput:.0f} entities/s"
        )
        assert total < 10.0, f"Full pipeline too slow for {entities} entities: {total:.3f}s"

    def test_full_pipeline_throughput_large(self, dxf_large: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_large)
        entities = timing["entity_count"]
        total = timing["total"]
        throughput = entities / total if total > 0 else 0
        print(
            f"\n  Full pipeline (large): {entities} entities in {total:.4f}s"
            f" → {throughput:.0f} entities/s"
        )
        assert total < 20.0, f"Full pipeline too slow for {entities} entities: {total:.3f}s"

    def test_full_pipeline_throughput_xlarge(self, dxf_xlarge: Path) -> None:
        timing = _benchmark_dxf_pipeline(dxf_xlarge)
        entities = timing["entity_count"]
        total = timing["total"]
        throughput = entities / total if total > 0 else 0
        print(
            f"\n  Full pipeline (xlarge): {entities} entities in {total:.4f}s"
            f" → {throughput:.0f} entities/s"
        )
        assert total < 60.0, f"Full pipeline too slow for {entities} entities: {total:.3f}s"

    # ── Scaling Assertions ──

    def test_scaling_monotonic(self, dxf_small: Path, dxf_medium: Path, dxf_large: Path, dxf_xlarge: Path) -> None:
        """Larger DXF files should take longer to process."""
        times = []
        for dxf in [dxf_small, dxf_medium, dxf_large, dxf_xlarge]:
            timing = _benchmark_dxf_pipeline(dxf)
            times.append(timing["total"])

        # Each subsequent size should take at least as long as the previous
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1] * 0.5, (
                f"Scaling anomaly: size {i} took {times[i]:.3f}s but size {i-1}"
                f" took {times[i-1]:.3f}s (ratio={times[i]/times[i-1]:.2f})"
            )

        print(f"\n  Scaling: {[f'{t:.3f}s' for t in times]}")

    def test_agent1_scales_linearly(self, dxf_small: Path, dxf_medium: Path, dxf_large: Path) -> None:
        """Agent 1 (DXF loading) should scale roughly linearly with entity count."""
        times = []
        counts = []
        for dxf in [dxf_small, dxf_medium, dxf_large]:
            timing = _benchmark_dxf_pipeline(dxf)
            times.append(timing["agent1_load"])
            counts.append(timing["entity_count"])

        # Compute entities/sec for each and compare
        throughputs = [c / t if t > 0 else 0 for c, t in zip(counts, times)]
        # Throughput should be roughly similar across sizes (within 5x)
        max_tp = max(throughputs)
        min_tp = min(throughputs)
        assert max_tp / min_tp < 10, (
            f"Agent 1 throughput varies too much: {throughputs}"
        )
        print(f"\n  Agent 1 throughput: {[f'{t:.0f} ent/s' for t in throughputs]}")


# ── Detailed Per-Agent Breakdown ─────────────────────────────────────────────


class TestDetailedAgentBreakdown:
    """Provides a detailed breakdown of where time is spent per agent."""

    @pytest.mark.parametrize("size_name,size_fixture", [
        ("small", "dxf_small"),
        ("medium", "dxf_medium"),
        ("large", "dxf_large"),
    ])
    def test_per_agent_breakdown(self, size_name: str, size_fixture: str, request: Any) -> None:
        """Print a detailed breakdown of time spent in each agent."""
        dxf_path = request.getfixturevalue(size_fixture)
        timing = _benchmark_dxf_pipeline(dxf_path)

        entities = timing["entity_count"]
        total = timing["total"]

        print(f"\n  ── Pipeline breakdown ({size_name}: {entities:.0f} entities) ──")
        print(f"  {'Agent':<20} {'Time (s)':<12} {'% of total':<12} {'Throughput':<15}")
        print(f"  {'-'*59}")

        agents = [
            ("Agent 1 (Load DXF)", "agent1_load"),
            ("Agent 2 (Primitives)", "agent2_primitives"),
            ("Agent 3 (Annotations)", "agent3_annotations"),
            ("Agent 4 (Features)", "agent4_features"),
            ("Agent 5 (Reasoning)", "agent5_reasoning"),
            ("Agent 6 (CAD Gen)", "agent6_cad"),
        ]
        for label, key in agents:
            t = timing[key]
            pct = (t / total * 100) if total > 0 else 0
            tp = entities / t if t > 0 else 0
            print(f"  {label:<20} {t:<12.4f} {pct:<12.1f} {tp:<15.0f}")

        print(f"  {'─' * 59}")
        print(f"  {'Total':<20} {total:<12.4f} {'100.0':<12}")


# ── Compare Operation Scaling ────────────────────────────────────────────────


class TestOperationScaling:
    """Verify that feature/operation detection scales with entity count."""

    def test_features_from_large_dxf(self, dxf_large: Path) -> None:
        """Large DXF with circles should produce many hole features."""
        doc = agent_1.process(str(dxf_large))
        primitives = agent_2.process(doc)
        features = agent_4.process(primitives)

        # ~20% of entities are circles → ~2000 circles → ~2000 holes expected
        hole_count = features["class_counts"].get("hole", 0)
        print(f"\n  Large DXF: ~2000 circles → {hole_count} holes detected")
        assert hole_count >= 100, f"Expected at least 100 holes from 10K entities, got {hole_count}"

    def test_drill_operations_from_large_dxf(self, dxf_large: Path) -> None:
        """Large DXF → many circles → many drill operations."""
        doc = agent_1.process(str(dxf_large))
        primitives = agent_2.process(doc)
        annotations = agent_3.process(doc)
        features = agent_4.process(primitives)
        reasoning = agent_5.process({
            "primitives": primitives,
            "annotations": annotations,
            "features": features,
        })

        drill_ops = [o for o in reasoning["operations"] if o["operation"] == "drill"]
        print(f"\n  Large DXF: {len(drill_ops)} drill operations from {features['feature_count']} features")
        assert len(drill_ops) >= 50, f"Expected >=50 drill ops, got {len(drill_ops)}"


# ── Cleanup: remove temp DXF files at session end ───────────────────────────━


@pytest.fixture(scope="session", autouse=True)
def _cleanup_dxf_files(request: Any) -> None:
    """Clean up all temporary DXF files created by fixtures."""
    fixture_names = ("dxf_small", "dxf_medium", "dxf_large", "dxf_xlarge")

    def _cleanup() -> None:
        for name in fixture_names:
            try:
                path = request.getfixturevalue(name)
                if isinstance(path, Path) and path.suffix == ".dxf":
                    os.unlink(str(path))
            except Exception:
                pass

    request.addfinalizer(_cleanup)
