"""Diff engine: compare two run reports and detect regressions."""

from __future__ import annotations

from rich.table import Table

from .models import DiffReport, GradedResult, RunReport


def compute_diff(
    baseline: RunReport,
    current: RunReport,
    threshold: float | None = None,
) -> DiffReport:
    """Compare per-failure-mode pass rates and find flipped questions.

    ``threshold`` is in percentage points. If set, ``regression_detected`` is
    True when any failure mode's pass rate drops by more than ``threshold``.
    """
    baseline_results = {r.question_id: r for r in baseline.results}
    current_results = {r.question_id: r for r in current.results}

    new_failures: list[GradedResult] = []
    new_passes: list[GradedResult] = []
    for qid, cur in current_results.items():
        base = baseline_results.get(qid)
        if base is None:
            continue
        if base.passed and not cur.passed:
            new_failures.append(cur)
        elif not base.passed and cur.passed:
            new_passes.append(cur)

    regressed_failure_modes: list[str] = []
    all_modes = set(baseline.by_failure_mode) | set(current.by_failure_mode)
    for mode in all_modes:
        base_rate = baseline.by_failure_mode.get(mode, {}).get("pass_rate", 0.0)
        cur_rate = current.by_failure_mode.get(mode, {}).get("pass_rate", 0.0)
        delta_pp = (cur_rate - base_rate) * 100.0
        if threshold is not None and delta_pp < -threshold:
            regressed_failure_modes.append(mode)

    overall_delta = current.pass_rate - baseline.pass_rate
    regression_detected = threshold is not None and len(regressed_failure_modes) > 0

    return DiffReport(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        overall_delta=round(overall_delta, 4),
        regressed_failure_modes=sorted(regressed_failure_modes),
        new_failures=new_failures,
        new_passes=new_passes,
        regression_detected=regression_detected,
    )


def format_diff_table(
    baseline: RunReport, current: RunReport, diff: DiffReport
) -> Table:
    """Build a Rich table: Failure Mode | Baseline | Current | Delta | Status."""
    table = Table(show_lines=False)
    table.add_column("Failure Mode")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status", justify="center")

    modes = sorted(set(baseline.by_failure_mode) | set(current.by_failure_mode))
    for mode in modes:
        base_rate = baseline.by_failure_mode.get(mode, {}).get("pass_rate", 0.0)
        cur_rate = current.by_failure_mode.get(mode, {}).get("pass_rate", 0.0)
        _add_row(table, mode, base_rate, cur_rate)

    _add_row(table, "OVERALL", baseline.pass_rate, current.pass_rate, bold=True)
    return table


def _add_row(
    table: Table, label: str, base_rate: float, cur_rate: float, bold: bool = False
) -> None:
    delta_pp = (cur_rate - base_rate) * 100.0
    if delta_pp > 1.0:
        status, style = "[green]\u2713[/green]", "green"
    elif delta_pp < -1.0:
        status, style = "[red]\u2193[/red]", "red"
    else:
        status, style = "[grey62]\u2192[/grey62]", "grey62"

    name = f"[bold]{label}[/bold]" if bold else label
    table.add_row(
        name,
        f"{base_rate * 100:.0f}%",
        f"{cur_rate * 100:.0f}%",
        f"[{style}]{delta_pp:+.0f}%[/{style}]",
        status,
    )
