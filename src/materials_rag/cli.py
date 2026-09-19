"""Public command line interface for Materials RAG Lab."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from materials_rag import __version__
from materials_rag.benchmark import RETRIEVAL_LADDER_METHODS, run_benchmark
from materials_rag.dataset import dataset_status, install_dataset
from materials_rag.models import models_status, prepare_models
from materials_rag.project import find_project_root
from materials_rag.reconstruction import reconstruct_full_corpus

app = typer.Typer(no_args_is_help=True, help="Materials RAG Lab public interface.")
data_app = typer.Typer(no_args_is_help=True, help="Install and inspect the companion dataset.")
models_app = typer.Typer(no_args_is_help=True, help="Prepare and inspect model/runtime requirements.")
results_app = typer.Typer(no_args_is_help=True, help="Inspect and verify published result artifacts.")
app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")
app.add_typer(results_app, name="results")


def _root() -> Path:
    return find_project_root()


def _emit(value: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))
    elif isinstance(value, dict):
        for key, item in value.items():
            typer.echo(f"{key}: {item}")
    else:
        typer.echo(value)


def _doctor_report(capability: str) -> dict[str, object]:
    try:
        root = _root()
        project_checkout = True
    except FileNotFoundError:
        root = Path.cwd().resolve()
        project_checkout = False
    data = dataset_status(root).to_dict()
    models = models_status().to_dict()
    core_pass = (
        project_checkout
        and (root / "data/manifests/agentic_rag_v0.1.0_experiment_freeze.json").exists()
        and (root / "schemas").exists()
    )
    structured = (root / "data/runtime/dataset/canonical/records/experiments.jsonl").exists()
    checks: dict[str, object] = {
        "package_version": __version__,
        "project_checkout": project_checkout,
        "dataset": data,
        "models": models,
        "frozen_manifest": (
            root / "data/manifests/agentic_rag_v0.1.0_experiment_freeze.json"
        ).exists(),
        "schemas": (root / "schemas").exists(),
        "structured_records": structured,
        "capabilities": {
            "core": "PASS" if core_pass else "FAIL",
            "dense": (
                "PASS"
                if data["outcome"] == "PASS" and models["embedding_available"]
                else "WARN"
            ),
            "hybrid": (
                "PASS"
                if data["outcome"] == "PASS" and models["embedding_available"]
                else "WARN"
            ),
            "reranking": (
                "PASS"
                if data["outcome"] == "PASS" and models["reranker_available"]
                else "WARN"
            ),
            "agentic": (
                "PASS"
                if data["retrieval_unit_count"] == 277
                and models["embedding_available"]
                and models["codex_available"]
                else "WARN"
            ),
            "scientific": "PASS" if structured and models["codex_available"] else "WARN",
        },
    }
    capabilities = checks["capabilities"]
    assert isinstance(capabilities, dict)
    if capability == "all":
        values = set(capabilities.values())
        checks["outcome"] = (
            "FAIL" if "FAIL" in values else "PASS" if values == {"PASS"} else "WARN"
        )
    elif capability not in capabilities:
        raise typer.BadParameter(
            "capability must be all, core, dense, hybrid, reranking, agentic, or scientific"
        )
    else:
        checks["outcome"] = capabilities[capability]
    checks["requested_capability"] = capability
    return checks


@app.command()
def doctor(
    capability: Annotated[str, typer.Option("--capability")] = "all",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Check repository, data, models, and runtime readiness by capability."""

    checks = _doctor_report(capability.lower().replace("_", "-"))
    _emit(checks, json_output)


@data_app.command("status")
def data_status(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Report whether 252-unit or reconstructed 277-unit data are installed."""

    _emit(dataset_status(_root()).to_dict(), json_output)


@data_app.command("install")
def data_install(
    source: Annotated[Path, typer.Argument(help="Dataset v0.1.0 ZIP or extracted root.")],
    check: Annotated[bool, typer.Option("--check")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Verify and install the redistributable companion dataset projection."""

    _emit(install_dataset(source, _root(), check_only=check), json_output)


@data_app.command("reconstruct")
def data_reconstruct(
    source_root: Annotated[Path, typer.Option("--source-root")],
    output: Annotated[Path, typer.Option("--output")] = Path("data/runtime"),
    check: Annotated[bool, typer.Option("--check")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Verify authorized sources and reconstruct the 277-unit retrieval corpus."""

    output_root = output if output.is_absolute() else _root() / output
    report = reconstruct_full_corpus(
        _root(), source_root, output_root, check_only=check
    )
    _emit(report, json_output)


@models_app.command("status")
def model_status(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Check the frozen embedding, reranker, and Codex requirements."""

    _emit(models_status().to_dict(), json_output)


@models_app.command("prepare")
def model_prepare(
    no_reranker: Annotated[bool, typer.Option("--no-reranker")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Download the declared Hugging Face model snapshots."""

    _emit(prepare_models(include_reranker=not no_reranker), json_output)


@app.command()
def retrieve(
    query: Annotated[str, typer.Option("--query")],
    method: Annotated[str, typer.Option("--method")] = "hybrid",
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=50)] = 10,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Execute one query with a named retrieval capability."""

    from materials_rag.api import MaterialsRAG

    rows = MaterialsRAG.from_dataset().retrieve(query, method=method, top_k=top_k)
    if json_output:
        _emit(rows, True)
    else:
        for row in rows:
            typer.echo(f"{row['rank']:>2}  {row['retrieval_unit_id']}")
            typer.echo(f"    {row['retrieval_text'][:220].replace(chr(10), ' ')}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Ask an arbitrary question using bounded Agentic RAG."""

    from materials_rag.api import MaterialsRAG

    result = MaterialsRAG.from_dataset(require_full=True).ask(question)
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if output:
        if output.exists():
            raise typer.BadParameter(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        typer.echo(output)
    else:
        typer.echo(text)


@app.command()
def demo(
    capability: Annotated[str, typer.Argument()],
    query: Annotated[str | None, typer.Option("--query")] = None,
) -> None:
    """Run a capability demonstration or inspect its frozen public summary."""

    mapping = {
        "data-preparation": "00_data_and_knowledge_preparation.py",
        "vanilla": "01_vanilla_dense.py",
        "reranking": "02_reranking.py",
        "hybrid-reranking": "02_reranking.py",
        "hybrid": "03_hybrid.py",
        "multi-query": "04_multi_query.py",
        "decomposition": "05_decomposition.py",
        "adaptive": "06_adaptive.py",
        "corrective": "07_corrective.py",
        "agentic": "08_agentic_retrieval.py",
        "scientific-computation": "09_scientific_computation.py",
        "verified-agentic": "10_verified_agentic_rag.py",
    }
    if capability not in mapping:
        raise typer.BadParameter(f"Unknown capability: {capability}")
    if query and capability in {
        "vanilla",
        "reranking",
        "hybrid",
        "hybrid-reranking",
        "multi-query",
        "decomposition",
    }:
        method = {
            "vanilla": "dense",
            "reranking": "dense-reranked",
            "hybrid-reranking": "hybrid-reranked",
        }.get(capability, capability)
        retrieve(query=query, method=method, top_k=10, json_output=False)
        return
    if query and capability == "agentic":
        ask(question=query, output=None)
        return
    subprocess.run([sys.executable, str(_root() / "experiments" / mapping[capability])], check=True)


@app.command("run", hidden=True)
def run_capability(
    capability: Annotated[str, typer.Argument()],
    query: Annotated[str | None, typer.Option("--query")] = None,
) -> None:
    """Compatibility alias for demo."""

    demo(capability=capability, query=query)


@results_app.command("verify")
def results_verify() -> None:
    """Verify compact published summaries against frozen public manifests."""

    subprocess.run(
        [sys.executable, str(_root() / "scripts/verify_public_results.py")],
        check=True,
        cwd=_root(),
    )


def _method_tuple(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    methods = tuple(
        item.strip().lower().replace("_", "-")
        for item in value.split(",")
        if item.strip()
    )
    if not methods:
        raise typer.BadParameter("--methods must contain at least one method")
    unsupported = set(methods) - set(RETRIEVAL_LADDER_METHODS)
    if unsupported:
        raise typer.BadParameter(f"Unsupported methods: {sorted(unsupported)}")
    return methods


@app.command()
def benchmark(
    suite: Annotated[str, typer.Argument()] = "retrieval-ladder",
    output: Annotated[Path, typer.Option("--output")] = Path("benchmark-output"),
    methods: Annotated[str | None, typer.Option("--methods")] = None,
) -> None:
    """Execute a benchmark suite and write new results outside frozen artifacts."""

    report = run_benchmark(
        _root(),
        output,
        suite=suite,
        methods=_method_tuple(methods),
    )
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


@app.command()
def reproduce(
    suite: Annotated[str, typer.Option("--suite")] = "retrieval-ladder",
    output: Annotated[Path, typer.Option("--output")] = Path("reproduction-output"),
) -> None:
    """Execute a fresh benchmark reproduction into a new output root."""

    report = run_benchmark(_root(), output, suite=suite)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
