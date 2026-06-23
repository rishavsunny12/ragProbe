"""RAGProbe command-line interface (Typer app)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import typer

# Ensure Unicode output (status glyphs, box-drawing tables) works on terminals
# whose default codepage is not UTF-8 (e.g. Windows cp1252), avoiding
# UnicodeEncodeError when printing characters like the check mark or arrows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

from .models import ALL_FAILURE_MODES, Chunk, Question, RunReport
from .storage import DEFAULT_DB_PATH, Storage

app = typer.Typer(
    add_completion=False,
    help="RAGProbe: find the questions your RAG pipeline will fail on, before your users do.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

DEFAULT_LLM = os.environ.get("RAGPROBE_DEFAULT_LLM", "openai/gpt-4o-mini")


# -- shared helpers ------------------------------------------------------


def _db_path(db: str | None) -> str:
    return db or os.environ.get("RAGPROBE_DB_PATH", DEFAULT_DB_PATH)


def _error(message: str, suggestion: str | None = None) -> None:
    body = message
    if suggestion:
        body += f"\n\n[bold]Suggested fix:[/bold] {suggestion}"
    err_console.print(Panel(body, title="[red]Error[/red]", border_style="red"))


def _fail(message: str, suggestion: str | None = None, code: int = 1) -> None:
    _error(message, suggestion)
    raise typer.Exit(code)


def _check_llm_configured(llm: str) -> None:
    """Verify the API key needed for the chosen LLM provider is present."""
    provider = llm.split("/", 1)[0].lower()
    needs = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    required = needs.get(provider)
    if required and not os.environ.get(required):
        _fail(
            "No LLM configured. Options:\n"
            "  1. Set OPENAI_API_KEY for cloud inference\n"
            "  2. Use --llm ollama/llama3 for local inference "
            "(run: ollama pull llama3 && ollama serve)",
        )


def _read_chunks_jsonl(path: Path) -> list[Chunk]:
    if not path.exists():
        _fail(f"Chunks file not found: {path}")
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.model_validate_json(line))
            except Exception as e:  # noqa: BLE001
                _fail(f"Invalid chunk on line {lineno} of {path}: {e}")
    if not chunks:
        _fail(f"No chunks found in {path}.")
    return chunks


def _read_questions_jsonl(path: Path) -> list[Question]:
    if not path.exists():
        _fail(f"Questions file not found: {path}")
    questions: list[Question] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(Question.model_validate_json(line))
            except Exception as e:  # noqa: BLE001
                _fail(f"Invalid question on line {lineno} of {path}: {e}")
    if not questions:
        _fail("No questions found. Run: ragprobe generate")
    return questions


def _write_questions_jsonl(path: Path, questions: list[Question]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(q.model_dump_json() + "\n")


def _load_graph_from_db(db_path: str, embedding_model: str):
    from .graph import ChunkGraph

    with Storage(db_path) as store:
        if store.chunk_count() == 0:
            _fail(
                "No indexed chunks found.",
                suggestion="Run: ragprobe index <chunks_file>",
            )
        chunks = store.load_chunks()
    graph = ChunkGraph(chunks, embedding_model=embedding_model)
    graph.build()  # embeddings already present -> only builds graph + centrality
    return graph


# -- index ---------------------------------------------------------------


@app.command()
def index(
    chunks_file: Path = typer.Argument(..., help="Path to chunks JSONL file."),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2", "--embedding-model"),
    db: str = typer.Option(None, "--db", help="SQLite DB path."),
    batch_size: int = typer.Option(32, "--batch-size"),
) -> None:
    """Embed chunks and build the topology graph."""
    from .graph import ChunkGraph

    db_path = _db_path(db)
    chunks = _read_chunks_jsonl(chunks_file)

    if len(chunks) < 10:
        console.print(
            "[yellow]Warning:[/yellow] Index at least 50 chunks for meaningful results."
        )

    graph = ChunkGraph(chunks, embedding_model=embedding_model, batch_size=batch_size)

    console.print(f"Embedding {len(chunks):,} chunks with [cyan]{embedding_model}[/cyan]...")
    try:
        model = graph.model
        texts = [c.text for c in chunks]
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding", total=len(chunks))
            for i in range(0, len(chunks), batch_size):
                batch = texts[i : i + batch_size]
                vectors = model.encode(
                    batch, convert_to_numpy=True, show_progress_bar=False
                )
                for chunk, vec in zip(chunks[i : i + batch_size], vectors):
                    chunk.embedding = [float(x) for x in vec]
                progress.update(task, advance=len(batch))
        graph.build()
    except Exception as e:  # noqa: BLE001
        _fail(f"Failed to embed/build graph: {e}")

    with Storage(db_path) as store:
        store.save_chunks(chunks)

    summary = graph.summary()
    table = Table(show_header=False, box=None)
    table.add_row("Chunks indexed", f"{summary['chunk_count']:,}")
    table.add_row("Connected components", str(summary["connected_components"]))
    table.add_row("Isolated chunks", str(summary["isolated_chunks"]))
    console.print(table)
    console.print(f"\n[green]\u2713[/green] Index written to {db_path}")
    console.print("  Next: ragprobe generate")


# -- generate ------------------------------------------------------------


@app.command()
def generate(
    output: Path = typer.Option(Path(".ragprobe/questions.jsonl"), "--output"),
    failure_modes: str = typer.Option(
        ",".join(ALL_FAILURE_MODES), "--failure-modes", help="Comma-separated."
    ),
    count: int = typer.Option(20, "--count", help="Questions per failure mode."),
    llm: str = typer.Option(DEFAULT_LLM, "--llm"),
    db: str = typer.Option(None, "--db"),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2", "--embedding-model"),
) -> None:
    """Generate adversarial questions from the indexed chunk graph."""
    from .generators import GENERATORS

    _check_llm_configured(llm)
    db_path = _db_path(db)

    modes = [m.strip() for m in failure_modes.split(",") if m.strip()]
    invalid = [m for m in modes if m not in ALL_FAILURE_MODES]
    if invalid:
        _fail(
            f"Unknown failure mode(s): {', '.join(invalid)}",
            suggestion=f"Valid modes: {', '.join(ALL_FAILURE_MODES)}",
        )

    graph = _load_graph_from_db(db_path, embedding_model)
    console.print(
        f"Generating adversarial questions from {len(graph.chunks):,} chunks...\n"
    )

    all_questions: list[Question] = []
    for mode in modes:
        gen = GENERATORS[mode](graph, llm)
        try:
            questions = gen.generate(count)
        except Exception as e:  # noqa: BLE001
            _fail(f"Generation failed for {mode}: {e}")
        # renumber ids so they are unique & stable across the full output
        for q in questions:
            all_questions.append(q)
        console.print(
            f"  [cyan]{mode:<12}[/cyan] {len(questions):>3}/{count} "
            f"({gen.discarded} discarded by self-verify)"
        )

    if not all_questions:
        _fail(
            "No questions were generated (all candidates discarded or no candidates found).",
            suggestion="Try a larger/denser corpus or a different --llm.",
        )

    _write_questions_jsonl(output, all_questions)
    with Storage(db_path) as store:
        store.save_questions(all_questions)

    console.print(
        f"\n[green]\u2713[/green] Generated {len(all_questions)} questions \u2192 {output}"
    )
    console.print(
        f"  Next: ragprobe run {output} --pipeline http://localhost:8000/query"
    )


# -- run -----------------------------------------------------------------


@app.command()
def run(
    questions_file: Path = typer.Argument(..., help="Path to questions JSONL file."),
    pipeline: str = typer.Option(..., "--pipeline", help="RAG pipeline URL (required)."),
    output: Path = typer.Option(None, "--output"),
    request_template: str = typer.Option(
        '{"query": "{{question}}"}', "--request-template"
    ),
    response_path: str = typer.Option("answer", "--response-path"),
    grader_llm: str = typer.Option(DEFAULT_LLM, "--grader-llm"),
    concurrency: int = typer.Option(3, "--concurrency"),
    timeout: float = typer.Option(
        120.0, "--timeout", help="HTTP timeout per pipeline call (seconds)."
    ),
    db: str = typer.Option(None, "--db"),
) -> None:
    """Call the RAG pipeline over HTTP and grade the answers."""
    from .runner import AnswerGrader, PipelineError, PipelineRunner, build_run_report, make_run_id

    _check_llm_configured(grader_llm)
    db_path = _db_path(db)
    questions = _read_questions_jsonl(questions_file)

    with Storage(db_path) as store:
        id_to_chunk = {c.id: c for c in store.load_chunks()}

    runner = PipelineRunner(pipeline, request_template, response_path, timeout=timeout)
    grader = AnswerGrader(grader_llm)

    console.print(f"Running {len(questions)} questions against [cyan]{pipeline}[/cyan]...")
    try:
        answers = asyncio.run(runner.run_all(questions, concurrency=concurrency))
    except PipelineError as e:
        _fail(str(e), suggestion="Check that your pipeline is running and the URL is correct.")

    answers_by_id = {a.question_id: a for a in answers}

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Grading", total=len(questions))
        for q in questions:
            answer = answers_by_id[q.id]
            source_chunks = [
                id_to_chunk[cid] for cid in q.source_chunk_ids if cid in id_to_chunk
            ]
            try:
                results.append(grader.grade(q, answer, source_chunks))
            except PipelineError as e:
                console.print(f"[yellow]Warning:[/yellow] grading failed for {q.id}: {e}")
            progress.update(task, advance=1)

    run_id = make_run_id()
    report = build_run_report(results, pipeline_url=pipeline, run_id=run_id)

    out_path = output or Path(f".ragprobe/runs/{run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    with Storage(db_path) as store:
        store.save_run(report)

    _print_run_summary(report)
    console.print(f"\n[green]\u2713[/green] Report written to {out_path}")


def _print_run_summary(report: RunReport) -> None:
    table = Table(title=f"Run {report.run_id}")
    table.add_column("Failure Mode")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Pass Rate", justify="right")
    for mode, stats in sorted(report.by_failure_mode.items()):
        table.add_row(
            mode,
            str(stats["passed"]),
            str(stats["total"]),
            f"{stats['pass_rate'] * 100:.0f}%",
        )
    table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold]{report.passed}[/bold]",
        f"[bold]{report.total_questions}[/bold]",
        f"[bold]{report.pass_rate * 100:.0f}%[/bold]",
    )
    console.print(table)


# -- diff ----------------------------------------------------------------


@app.command()
def diff(
    baseline_file: Path = typer.Argument(..., help="Baseline run report JSON."),
    current_file: Path = typer.Argument(..., help="Current run report JSON."),
    fail_on_regression: float = typer.Option(
        None,
        "--fail-on-regression",
        help="Exit 1 if any failure mode drops more than N percentage points.",
    ),
    output: Path = typer.Option(None, "--output", help="Optional path to write the diff JSON."),
) -> None:
    """Compare two run reports and detect regressions."""
    from .diff import compute_diff, format_diff_table

    baseline = _load_report(baseline_file)
    current = _load_report(current_file)

    diff_report = compute_diff(baseline, current, threshold=fail_on_regression)

    console.print(
        f"[bold]Pipeline Regression Report[/bold]\n"
        f"baseline: {baseline.run_id} \u2192 current: {current.run_id}\n"
    )
    console.print(format_diff_table(baseline, current, diff_report))

    if diff_report.new_failures:
        console.print(f"\n[red]New failures ({len(diff_report.new_failures)}):[/red]")
        for r in diff_report.new_failures:
            console.print(f"  {r.question_id} [{r.failure_mode}] \"{r.question}\"")
    if diff_report.new_passes:
        console.print(f"\n[green]New passes ({len(diff_report.new_passes)}):[/green]")
        for r in diff_report.new_passes:
            console.print(f"  {r.question_id} [{r.failure_mode}] \"{r.question}\"")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(diff_report.model_dump_json(indent=2), encoding="utf-8")

    if diff_report.regression_detected:
        console.print(
            f"\n[red]EXIT CODE: 1[/red] (regression threshold of "
            f"{fail_on_regression} pp breached: "
            f"{', '.join(diff_report.regressed_failure_modes)})"
        )
        raise typer.Exit(1)
    console.print("\n[green]EXIT CODE: 0[/green] (no regression threshold breached)")


def _load_report(path: Path) -> RunReport:
    if not path.exists():
        _fail(f"Run report not found: {path}")
    try:
        return RunReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        _fail(f"Invalid run report {path}: {e}")


# -- calibrate -----------------------------------------------------------


@app.command()
def calibrate(
    pipeline: str = typer.Option(..., "--pipeline", help="RAG pipeline URL (required)."),
    sample_size: int = typer.Option(30, "--sample-size"),
    llm: str = typer.Option(DEFAULT_LLM, "--llm"),
    grader_llm: str = typer.Option(DEFAULT_LLM, "--grader-llm"),
    request_template: str = typer.Option('{"query": "{{question}}"}', "--request-template"),
    response_path: str = typer.Option("answer", "--response-path"),
    concurrency: int = typer.Option(3, "--concurrency"),
    db: str = typer.Option(None, "--db"),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2", "--embedding-model"),
) -> None:
    """Prove topology-aware questions are harder than random ones."""
    from .calibrate import run_calibration

    _check_llm_configured(llm)
    db_path = _db_path(db)
    graph = _load_graph_from_db(db_path, embedding_model)

    console.print("Running calibration (this generates and grades 3 question sets)...\n")
    try:
        report = run_calibration(
            graph=graph,
            llm_model=llm,
            pipeline_url=pipeline,
            request_template=request_template,
            response_path=response_path,
            grader_llm=grader_llm,
            sample_size=sample_size,
            concurrency=concurrency,
        )
    except Exception as e:  # noqa: BLE001
        _fail(f"Calibration failed: {e}")

    table = Table(title="Calibration Report")
    table.add_column("Question Set")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Samples", justify="right")
    table.add_row(
        "baseline random",
        f"{report.baseline_pass_rate * 100:.0f}%",
        str(report.sample_sizes.get("baseline", 0)),
    )
    table.add_row(
        "random multi-question",
        f"{report.random_multi_pass_rate * 100:.0f}%",
        str(report.sample_sizes.get("random_multi", 0)),
    )
    table.add_row(
        "RAGProbe topology-hard",
        f"{report.topology_hard_pass_rate * 100:.0f}%",
        str(report.sample_sizes.get("topology_hard", 0)),
    )
    console.print(table)
    console.print(
        f"\nDifficulty delta (easiest random \u2212 topology-hard): "
        f"[bold]{report.difficulty_delta_pp:+.1f} pp[/bold]"
    )

    if report.warning:
        console.print(
            Panel(
                "Calibration warning: topology-hard questions are NOT meaningfully "
                "harder than random (delta < 10 pp). The corpus may be too small or "
                "uniform for topology-aware generation to add value.",
                title="[yellow]Calibration Warning[/yellow]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            "[green]\u2713[/green] Topology-aware questions are measurably harder."
        )


if __name__ == "__main__":  # pragma: no cover
    app()
