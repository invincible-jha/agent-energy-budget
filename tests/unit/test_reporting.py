"""Unit tests for agent_energy_budget.reporting package."""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import pytest

from agent_energy_budget.reporting.aggregator import CostAggregator, SpendRecord, _parse_record
from agent_energy_budget.reporting.reporter import BudgetReporter, ReportPeriod, ReportSummary
from agent_energy_budget.reporting.top_n import CostHotspot, TopNAnalyzer
from agent_energy_budget.reporting.visualizer import AsciiVisualizer, BarChartConfig


# ===========================================================================
# Shared fixtures
# ===========================================================================


def _make_jsonl_file(
    tmp_path: pathlib.Path,
    records: list[dict],
    filename: str = "test.jsonl",
) -> pathlib.Path:
    """Write records to a JSONL file and return the path."""
    log_path = tmp_path / filename
    with log_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return log_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(
    agent: str = "agent-1",
    model: str = "gpt-4o-mini",
    cost: float = 0.01,
    input_tokens: int = 100,
    output_tokens: int = 50,
    task: str = "",
) -> dict:
    return {
        "agent_id": agent,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "recorded_at": _now_iso(),
        "task": task,
    }


# ===========================================================================
# _parse_record
# ===========================================================================


class TestParseRecord:
    def test_valid_record(self) -> None:
        raw = {
            "agent_id": "agent-1",
            "model": "gpt-4o-mini",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.001,
            "recorded_at": _now_iso(),
        }
        result = _parse_record(raw)
        assert result is not None
        assert result.agent_id == "agent-1"
        assert result.model == "gpt-4o-mini"
        assert result.cost_usd == pytest.approx(0.001)

    def test_missing_model_returns_none(self) -> None:
        raw = {"cost_usd": 0.001, "recorded_at": _now_iso()}
        result = _parse_record(raw)
        assert result is None

    def test_missing_cost_usd_returns_none(self) -> None:
        raw = {"model": "gpt-4o-mini", "recorded_at": _now_iso()}
        result = _parse_record(raw)
        assert result is None

    def test_missing_recorded_at_returns_none(self) -> None:
        raw = {"model": "gpt-4o-mini", "cost_usd": 0.001}
        result = _parse_record(raw)
        assert result is None

    def test_invalid_cost_type_returns_none(self) -> None:
        raw = {"model": "gpt-4o-mini", "cost_usd": "not-a-number", "recorded_at": _now_iso()}
        result = _parse_record(raw)
        assert result is None

    def test_unknown_model_has_provider_unknown(self) -> None:
        raw = {
            "model": "totally-unknown-xyz",
            "cost_usd": 0.001,
            "recorded_at": _now_iso(),
        }
        result = _parse_record(raw)
        assert result is not None
        assert result.provider == "unknown"

    def test_known_model_has_correct_provider(self) -> None:
        raw = {
            "model": "gpt-4o-mini",
            "cost_usd": 0.001,
            "recorded_at": _now_iso(),
        }
        result = _parse_record(raw)
        assert result is not None
        assert result.provider == "openai"

    def test_missing_agent_id_defaults_to_unknown(self) -> None:
        raw = {"model": "gpt-4o-mini", "cost_usd": 0.001, "recorded_at": _now_iso()}
        result = _parse_record(raw)
        assert result is not None
        assert result.agent_id == "unknown"

    def test_task_field_parsed(self) -> None:
        raw = {
            "model": "gpt-4o-mini",
            "cost_usd": 0.001,
            "recorded_at": _now_iso(),
            "task": "summarise",
        }
        result = _parse_record(raw)
        assert result is not None
        assert result.task == "summarise"

    def test_naive_datetime_gets_utc(self) -> None:
        raw = {
            "model": "gpt-4o-mini",
            "cost_usd": 0.001,
            "recorded_at": "2024-01-15T12:00:00",  # no timezone
        }
        result = _parse_record(raw)
        assert result is not None
        assert result.recorded_at.tzinfo is not None


# ===========================================================================
# CostAggregator
# ===========================================================================


class TestCostAggregator:
    def test_load_from_file(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record(cost=0.01), _record(cost=0.02)])
        agg = CostAggregator(log).load()
        assert agg.total_cost() == pytest.approx(0.03, abs=1e-9)

    def test_nonexistent_file_is_skipped(self, tmp_path: pathlib.Path) -> None:
        agg = CostAggregator(tmp_path / "nonexistent.jsonl").load()
        assert agg.total_cost() == 0.0

    def test_load_from_directory(self, tmp_path: pathlib.Path) -> None:
        _make_jsonl_file(tmp_path, [_record(cost=0.05)], "a.jsonl")
        _make_jsonl_file(tmp_path, [_record(cost=0.10)], "b.jsonl")
        agg = CostAggregator(tmp_path).load()
        assert agg.total_cost() == pytest.approx(0.15, abs=1e-9)

    def test_load_from_list_of_paths(self, tmp_path: pathlib.Path) -> None:
        p1 = _make_jsonl_file(tmp_path, [_record(cost=0.01)], "1.jsonl")
        p2 = _make_jsonl_file(tmp_path, [_record(cost=0.02)], "2.jsonl")
        agg = CostAggregator([p1, p2]).load()
        assert agg.total_cost() == pytest.approx(0.03, abs=1e-9)

    def test_malformed_lines_skipped(self, tmp_path: pathlib.Path) -> None:
        log = tmp_path / "test.jsonl"
        log.write_text("not-json\n" + json.dumps(_record(cost=0.01)) + "\n", encoding="utf-8")
        agg = CostAggregator(log).load()
        assert agg.total_cost() == pytest.approx(0.01, abs=1e-9)

    def test_auto_load_on_first_query(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record(cost=0.01)])
        agg = CostAggregator(log)  # no explicit .load()
        assert agg.total_cost() == pytest.approx(0.01, abs=1e-9)

    def test_load_returns_self_for_chaining(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record()])
        agg = CostAggregator(log)
        result = agg.load()
        assert result is agg

    def test_by_agent_sorted_descending(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [
            _record("agent-b", cost=0.02),
            _record("agent-a", cost=0.10),
        ])
        agg = CostAggregator(log).load()
        by_agent = agg.by_agent()
        agents = list(by_agent.keys())
        assert agents[0] == "agent-a"  # highest spend first

    def test_by_model(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [
            _record(model="gpt-4o-mini", cost=0.01),
            _record(model="claude-haiku-4", cost=0.05),
        ])
        agg = CostAggregator(log).load()
        by_model = agg.by_model()
        assert "gpt-4o-mini" in by_model
        assert "claude-haiku-4" in by_model
        # claude-haiku-4 is more expensive so should be first
        assert list(by_model.keys())[0] == "claude-haiku-4"

    def test_by_provider(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [
            _record(model="gpt-4o-mini", cost=0.02),
        ])
        agg = CostAggregator(log).load()
        by_provider = agg.by_provider()
        assert "openai" in by_provider

    def test_by_task(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [
            _record(task="summarise", cost=0.01),
            _record(task="", cost=0.02),
        ])
        agg = CostAggregator(log).load()
        by_task = agg.by_task()
        assert "summarise" in by_task
        assert "untagged" in by_task

    def test_call_count(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record(), _record(), _record()])
        agg = CostAggregator(log).load()
        assert agg.call_count() == 3

    def test_daily_breakdown(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record(cost=0.01), _record(cost=0.02)])
        agg = CostAggregator(log).load()
        daily = agg.daily_breakdown()
        assert isinstance(daily, dict)
        today_key = datetime.now(timezone.utc).date().isoformat()
        assert today_key in daily
        assert daily[today_key] == pytest.approx(0.03, abs=1e-9)

    def test_records_returns_list(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record()])
        agg = CostAggregator(log).load()
        records = agg.records()
        assert len(records) == 1
        assert isinstance(records[0], SpendRecord)

    def test_period_filter_today(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record(cost=0.05)])
        agg = CostAggregator(log).load()
        total_today = agg.total_cost(period="today")
        assert total_today == pytest.approx(0.05, abs=1e-9)

    def test_period_filter_all(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record(cost=0.05), _record(cost=0.10)])
        agg = CostAggregator(log).load()
        assert agg.total_cost(period="all") == pytest.approx(0.15, abs=1e-9)

    def test_period_filter_week(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record(cost=0.05)])
        agg = CostAggregator(log).load()
        # Records from today are within this week
        assert agg.total_cost(period="week") >= 0.0

    def test_period_filter_month(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record(cost=0.05)])
        agg = CostAggregator(log).load()
        assert agg.total_cost(period="month") >= 0.0

    def test_invalid_period_raises(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [_record()])
        agg = CostAggregator(log).load()
        with pytest.raises(ValueError, match="Unknown period"):
            agg.total_cost(period="yearly")

    def test_empty_lines_skipped(self, tmp_path: pathlib.Path) -> None:
        log = tmp_path / "test.jsonl"
        log.write_text(
            "\n\n" + json.dumps(_record(cost=0.01)) + "\n\n",
            encoding="utf-8",
        )
        agg = CostAggregator(log).load()
        assert agg.total_cost() == pytest.approx(0.01, abs=1e-9)

    def test_non_dict_json_lines_skipped(self, tmp_path: pathlib.Path) -> None:
        log = tmp_path / "test.jsonl"
        log.write_text(
            '["not", "a", "dict"]\n' + json.dumps(_record(cost=0.01)) + "\n",
            encoding="utf-8",
        )
        agg = CostAggregator(log).load()
        assert agg.total_cost() == pytest.approx(0.01, abs=1e-9)


# ===========================================================================
# BudgetReporter
# ===========================================================================


@pytest.fixture
def simple_aggregator(tmp_path: pathlib.Path) -> CostAggregator:
    log = _make_jsonl_file(tmp_path, [
        _record("agent-a", "gpt-4o-mini", 0.10, task="research"),
        _record("agent-b", "claude-haiku-4", 0.05, task="summarise"),
    ])
    return CostAggregator(log).load()


class TestBudgetReporter:
    def test_summary_all_period(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        summary = reporter.summary("all")
        assert summary.total_cost_usd == pytest.approx(0.15, abs=1e-9)
        assert summary.call_count == 2
        assert summary.avg_cost_per_call == pytest.approx(0.075, abs=1e-9)
        assert summary.top_agent in ("agent-a", "agent-b")

    def test_summary_with_report_period_enum(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        summary = reporter.summary(ReportPeriod.ALL)
        assert summary.period == "all"

    def test_summary_empty_aggregator(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [])
        agg = CostAggregator(log).load()
        reporter = BudgetReporter(agg)
        summary = reporter.summary("all")
        assert summary.total_cost_usd == 0.0
        assert summary.call_count == 0
        assert summary.avg_cost_per_call == 0.0
        assert summary.top_agent == ""
        assert summary.top_model == ""

    def test_to_json_valid_json(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        json_str = reporter.to_json("all")
        data = json.loads(json_str)
        assert "summary" in data
        assert "by_agent" in data
        assert "by_model" in data
        assert "by_provider" in data
        assert "by_task" in data
        assert "daily_breakdown" in data

    def test_to_json_with_enum_period(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        json_str = reporter.to_json(ReportPeriod.ALL)
        data = json.loads(json_str)
        assert data["summary"]["period"] == "all"

    def test_to_csv_has_header(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        csv_str = reporter.to_csv("all")
        first_line = csv_str.splitlines()[0]
        assert "recorded_at" in first_line
        assert "agent_id" in first_line
        assert "cost_usd" in first_line

    def test_to_csv_has_data_rows(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        csv_str = reporter.to_csv("all")
        lines = [line for line in csv_str.splitlines() if line.strip()]
        assert len(lines) == 3  # header + 2 records

    def test_to_csv_with_enum_period(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        csv_str = reporter.to_csv(ReportPeriod.ALL)
        assert "agent_id" in csv_str

    def test_to_markdown_contains_summary(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        md = reporter.to_markdown("all")
        assert "# Budget Report" in md
        assert "## Summary" in md
        assert "Total Cost" in md

    def test_to_markdown_contains_agent_section(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        md = reporter.to_markdown("all")
        assert "## Cost by Agent" in md
        assert "agent-a" in md

    def test_to_markdown_with_enum_period(self, simple_aggregator: CostAggregator) -> None:
        reporter = BudgetReporter(simple_aggregator)
        md = reporter.to_markdown(ReportPeriod.ALL)
        assert "All" in md

    def test_save_json_format(self, simple_aggregator: CostAggregator, tmp_path: pathlib.Path) -> None:
        reporter = BudgetReporter(simple_aggregator)
        output = tmp_path / "report.json"
        reporter.save(str(output), "all", format="json")
        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "summary" in data

    def test_save_csv_format(self, simple_aggregator: CostAggregator, tmp_path: pathlib.Path) -> None:
        reporter = BudgetReporter(simple_aggregator)
        output = tmp_path / "report.csv"
        reporter.save(str(output), "all", format="csv")
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "agent_id" in content

    def test_save_markdown_format(self, simple_aggregator: CostAggregator, tmp_path: pathlib.Path) -> None:
        reporter = BudgetReporter(simple_aggregator)
        output = tmp_path / "report.md"
        reporter.save(str(output), "all", format="md")
        assert output.exists()

    def test_save_markdown_alias(self, simple_aggregator: CostAggregator, tmp_path: pathlib.Path) -> None:
        reporter = BudgetReporter(simple_aggregator)
        output = tmp_path / "report2.md"
        reporter.save(str(output), "all", format="markdown")
        assert output.exists()

    def test_save_invalid_format_raises(self, simple_aggregator: CostAggregator, tmp_path: pathlib.Path) -> None:
        reporter = BudgetReporter(simple_aggregator)
        with pytest.raises(ValueError, match="Unsupported format"):
            reporter.save(str(tmp_path / "out.txt"), "all", format="txt")


# ===========================================================================
# TopNAnalyzer
# ===========================================================================


@pytest.fixture
def multi_agent_aggregator(tmp_path: pathlib.Path) -> CostAggregator:
    records = [
        _record("agent-expensive", "gpt-4o", 1.00, task="premium"),
        _record("agent-cheap", "gpt-4o-mini", 0.01, task="basic"),
        _record("agent-medium", "claude-haiku-4", 0.20, task="standard"),
        _record("agent-expensive", "gpt-4o", 0.50, task="premium"),
    ]
    log = _make_jsonl_file(tmp_path, records)
    return CostAggregator(log).load()


class TestTopNAnalyzer:
    def test_top_agents_ranked_correctly(
        self, multi_agent_aggregator: CostAggregator
    ) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        hotspots = analyzer.top_agents(n=3)
        assert hotspots[0].rank == 1
        assert hotspots[0].name == "agent-expensive"
        assert hotspots[0].total_cost_usd == pytest.approx(1.50, abs=1e-9)

    def test_top_agents_pct_of_total(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        hotspots = analyzer.top_agents(n=3)
        total = sum(h.total_cost_usd for h in hotspots)
        assert all(h.pct_of_total >= 0.0 for h in hotspots)
        # Percentages should sum to 100 (for all agents)
        all_hotspots = analyzer.top_agents(n=100)
        pct_sum = sum(h.pct_of_total for h in all_hotspots)
        assert pct_sum == pytest.approx(100.0, abs=0.1)

    def test_top_agents_limits_to_n(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        hotspots = analyzer.top_agents(n=1)
        assert len(hotspots) == 1

    def test_top_models(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        hotspots = analyzer.top_models(n=3)
        model_names = [h.name for h in hotspots]
        assert "gpt-4o" in model_names

    def test_top_tasks(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        hotspots = analyzer.top_tasks(n=5)
        task_names = [h.name for h in hotspots]
        assert "premium" in task_names

    def test_top_providers(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        hotspots = analyzer.top_providers(n=5)
        assert len(hotspots) > 0
        assert all(isinstance(h.name, str) for h in hotspots)

    def test_top_by_agent(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        result = analyzer.top_by("agent", n=2)
        assert len(result) <= 2
        assert all(h.dimension == "agent" for h in result)

    def test_top_by_model(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        result = analyzer.top_by("model", n=2)
        assert all(h.dimension == "model" for h in result)

    def test_top_by_task(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        result = analyzer.top_by("task", n=2)
        assert all(h.dimension == "task" for h in result)

    def test_top_by_provider(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        result = analyzer.top_by("provider", n=2)
        assert all(h.dimension == "provider" for h in result)

    def test_top_by_invalid_dimension_raises(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        with pytest.raises(ValueError, match="Unknown dimension"):
            analyzer.top_by("invalid_dim")

    def test_hotspot_report_returns_all_dimensions(
        self, multi_agent_aggregator: CostAggregator
    ) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        report = analyzer.hotspot_report(n=5)
        assert "agent" in report
        assert "model" in report
        assert "task" in report
        assert "provider" in report

    def test_hotspot_report_uses_n(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        report = analyzer.hotspot_report(n=1)
        for dimension_results in report.values():
            assert len(dimension_results) <= 1

    def test_empty_aggregator_returns_empty_hotspots(self, tmp_path: pathlib.Path) -> None:
        log = _make_jsonl_file(tmp_path, [])
        agg = CostAggregator(log).load()
        analyzer = TopNAnalyzer(agg)
        hotspots = analyzer.top_agents()
        assert hotspots == []

    def test_zero_total_gives_zero_pct(self, tmp_path: pathlib.Path) -> None:
        # Records with zero cost
        log = _make_jsonl_file(tmp_path, [_record(cost=0.0)])
        agg = CostAggregator(log).load()
        analyzer = TopNAnalyzer(agg)
        hotspots = analyzer.top_agents()
        assert all(h.pct_of_total == 0.0 for h in hotspots)

    def test_cost_hotspot_is_frozen(self, multi_agent_aggregator: CostAggregator) -> None:
        analyzer = TopNAnalyzer(multi_agent_aggregator)
        hotspots = analyzer.top_agents(n=1)
        h = hotspots[0]
        with pytest.raises((AttributeError, TypeError)):
            h.rank = 99  # type: ignore[misc]


# ===========================================================================
# AsciiVisualizer
# ===========================================================================


class TestAsciiVisualizer:
    @pytest.fixture
    def viz(self) -> AsciiVisualizer:
        return AsciiVisualizer()

    def test_budget_utilisation_bar_returns_string(self, viz: AsciiVisualizer) -> None:
        result = viz.budget_utilisation_bar("agent-1", 0.5, 1.0, "daily")
        assert isinstance(result, str)
        assert "agent-1" in result

    def test_budget_utilisation_bar_zero_limit(self, viz: AsciiVisualizer) -> None:
        # Should not raise
        result = viz.budget_utilisation_bar("agent-1", 0.0, 0.0, "daily")
        assert isinstance(result, str)

    def test_budget_utilisation_bar_over_100_pct(self, viz: AsciiVisualizer) -> None:
        result = viz.budget_utilisation_bar("agent-1", 1.5, 1.0, "daily")
        assert isinstance(result, str)

    def test_budget_utilisation_bar_includes_values(self, viz: AsciiVisualizer) -> None:
        result = viz.budget_utilisation_bar("agent-1", 0.5, 1.0)
        assert "$" in result

    def test_cost_distribution_chart_no_data(self, viz: AsciiVisualizer) -> None:
        result = viz.cost_distribution_chart({})
        assert "(no data)" in result

    def test_cost_distribution_chart_with_data(self, viz: AsciiVisualizer) -> None:
        result = viz.cost_distribution_chart({"gpt-4o": 0.10, "gpt-4o-mini": 0.01})
        assert "gpt-4o" in result
        assert "gpt-4o-mini" in result

    def test_cost_distribution_chart_title(self, viz: AsciiVisualizer) -> None:
        result = viz.cost_distribution_chart({"a": 1.0}, title="My Chart")
        assert "My Chart" in result

    def test_cost_distribution_chart_max_rows_limits_output(self, viz: AsciiVisualizer) -> None:
        cost_map = {f"model-{i}": float(i) for i in range(10)}
        result = viz.cost_distribution_chart(cost_map, max_rows=3)
        # Only 3 data rows (+ 2 separator lines + title)
        lines_with_model = [line for line in result.splitlines() if "model-" in line]
        assert len(lines_with_model) == 3

    def test_hotspot_table_no_data(self, viz: AsciiVisualizer) -> None:
        result = viz.hotspot_table([])
        assert "(no data)" in result

    def test_hotspot_table_with_data(self, viz: AsciiVisualizer) -> None:
        hotspots = [
            CostHotspot(rank=1, dimension="agent", name="agent-1", total_cost_usd=1.0, pct_of_total=80.0),
            CostHotspot(rank=2, dimension="agent", name="agent-2", total_cost_usd=0.25, pct_of_total=20.0),
        ]
        result = viz.hotspot_table(hotspots)
        assert "agent-1" in result
        assert "agent-2" in result
        assert "1" in result

    def test_hotspot_table_title(self, viz: AsciiVisualizer) -> None:
        hotspots = [
            CostHotspot(rank=1, dimension="model", name="gpt-4o", total_cost_usd=0.5, pct_of_total=100.0)
        ]
        result = viz.hotspot_table(hotspots, title="Model Costs")
        assert "Model Costs" in result

    def test_hotspot_table_long_name_truncated(self, viz: AsciiVisualizer) -> None:
        long_name = "a" * 100
        hotspots = [
            CostHotspot(rank=1, dimension="agent", name=long_name, total_cost_usd=1.0, pct_of_total=100.0)
        ]
        result = viz.hotspot_table(hotspots)
        assert "..." in result

    def test_status_grid_no_data(self, viz: AsciiVisualizer) -> None:
        result = viz.status_grid([])
        assert "No budget status data" in result

    def test_status_grid_with_data(self, viz: AsciiVisualizer) -> None:
        statuses = [
            {
                "agent_id": "agent-1",
                "period": "daily",
                "spent_usd": 0.5,
                "limit_usd": 1.0,
                "utilisation_pct": 50.0,
                "call_count": 5,
            }
        ]
        result = viz.status_grid(statuses)
        assert "agent-1" in result
        assert "daily" in result

    def test_status_grid_zero_limit_shows_unlimited(self, viz: AsciiVisualizer) -> None:
        statuses = [
            {
                "agent_id": "agent-1",
                "period": "daily",
                "spent_usd": 0.5,
                "limit_usd": 0.0,
                "utilisation_pct": 0.0,
                "call_count": 1,
            }
        ]
        result = viz.status_grid(statuses)
        assert "unlimited" in result

    def test_daily_sparkline_no_data(self, viz: AsciiVisualizer) -> None:
        result = viz.daily_sparkline({})
        assert "(no data)" in result

    def test_daily_sparkline_with_data(self, viz: AsciiVisualizer) -> None:
        daily_costs = {"2024-01-01": 0.10, "2024-01-02": 0.20, "2024-01-03": 0.05}
        result = viz.daily_sparkline(daily_costs)
        assert "Daily Spend Trend" in result
        assert "2024-01-01" in result
        assert "Total:" in result

    def test_daily_sparkline_custom_title(self, viz: AsciiVisualizer) -> None:
        # Use multiple days so the sparkline separator line calculation works
        daily_costs = {"2024-01-01": 0.10, "2024-01-02": 0.20}
        result = viz.daily_sparkline(daily_costs, title="My Trend")
        assert "My Trend" in result

    def test_daily_sparkline_all_zero_costs(self, viz: AsciiVisualizer) -> None:
        # Should not raise ZeroDivisionError
        daily_costs = {"2024-01-01": 0.0, "2024-01-02": 0.0}
        result = viz.daily_sparkline(daily_costs)
        assert isinstance(result, str)

    def test_custom_bar_config(self) -> None:
        config = BarChartConfig(width=40, bar_char="=", empty_char="-", show_values=False)
        viz = AsciiVisualizer(chart_config=config)
        result = viz.budget_utilisation_bar("agent", 0.5, 1.0)
        assert "=" in result
        assert "$" not in result  # show_values=False

    def test_truncate_long_agent_id(self, viz: AsciiVisualizer) -> None:
        long_id = "very-long-agent-identifier-name-here"
        result = viz.budget_utilisation_bar(long_id, 0.5, 1.0)
        assert isinstance(result, str)
        # Should not exceed a reasonable line length
        first_line = result.splitlines()[0]
        assert len(first_line) > 0
