"""Command line interface.

    clinicflow run document.txt            # process one document
    clinicflow batch data/                 # process a directory
    clinicflow demo                        # run the bundled sample set
    clinicflow review batch.jsonl          # the human review queue
    clinicflow redact document.txt         # show what a model would receive
    clinicflow kb "lumbar mri"             # query the policy knowledge base

Exit codes: 0 all complete, 1 something needs review, 2 something was
rejected, 3 usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents.retriever import KnowledgeBase
from .guardrails import GuardrailConfig, redact_phi, screen_document
from .llm import build_provider
from .orchestrator import Workflow, WorkflowConfig, escalations, summarize_batch
from .types import Decision, Document, Severity, WorkflowState

EXIT_OK, EXIT_REVIEW, EXIT_REJECTED, EXIT_USAGE = 0, 1, 2, 3

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _workflow(args: argparse.Namespace) -> Workflow:
    guardrails = GuardrailConfig()
    if getattr(args, "no_redact", False):
        guardrails.redact_phi = False
    if getattr(args, "min_confidence", None) is not None:
        guardrails.min_field_confidence = args.min_confidence

    config = WorkflowConfig(
        guardrails=guardrails,
        audit_log_path=getattr(args, "audit_log", None),
    )
    provider_kwargs = {}
    if args.provider in {"ollama", "openai-compat", "vllm"}:
        provider_kwargs["model"] = args.model
    provider = build_provider(args.provider, **provider_kwargs)

    kb_path = getattr(args, "knowledge_base", None)
    knowledge_base = KnowledgeBase.from_file(kb_path) if kb_path else None
    return Workflow(config=config, provider=provider, knowledge_base=knowledge_base)


def _exit_code(states: list[WorkflowState]) -> int:
    if any(s.decision in (Decision.REJECTED, Decision.FAILED) for s in states):
        return EXIT_REJECTED
    if any(s.decision is Decision.NEEDS_REVIEW for s in states):
        return EXIT_REVIEW
    return EXIT_OK


def _render(state: WorkflowState, show_summary: bool = True) -> str:
    marker = {
        Decision.COMPLETE: "COMPLETE",
        Decision.NEEDS_REVIEW: "REVIEW",
        Decision.REJECTED: "REJECTED",
        Decision.FAILED: "FAILED",
    }[state.decision]

    lines = [
        f"[{marker}] {state.document.doc_id}  type={state.document.doc_type.value}  "
        f"fields={len(state.fields)}  confidence={state.mean_confidence:.2f}  "
        f"{state.duration_ms:.1f} ms"
    ]

    if state.review_reasons:
        lines.append("  review reasons:")
        for reason in state.review_reasons:
            lines.append(f"    - {reason}")

    flagged = [i for i in state.issues if i.severity >= Severity.WARNING]
    if flagged:
        lines.append("  issues:")
        for issue in flagged:
            block = " [BLOCKING]" if issue.blocking else ""
            lines.append(f"    {issue.code} {issue.severity.value:<9}{block} {issue.message}")

    lines.append("  agent trace:")
    for step in state.trace:
        lines.append(
            f"    {step['agent']:<12} {step['elapsed_ms']:>7.2f} ms  "
            f"conf={step['confidence']:.2f}  fields={step['fields']}  issues={step['issues']}"
        )

    if show_summary:
        summary = state.result_for("summarizer")
        if summary is not None:
            text = summary.output.get("summary_for_review") or summary.output.get("summary", "")
            if text:
                lines.append("  summary:")
                lines.extend(f"    {line}" for line in text.splitlines())
    return "\n".join(lines)


def _load_documents(path: Path) -> list[Document]:
    if path.is_dir():
        return [
            Document(text=p.read_text(encoding="utf-8"), doc_id=p.stem, source=str(p))
            for p in sorted(path.glob("*.txt"))
        ]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            Document(text=item["text"], doc_id=item.get("doc_id", f"doc-{i}"))
            for i, item in enumerate(payload)
        ]
    return [Document(text=path.read_text(encoding="utf-8"), doc_id=path.stem, source=str(path))]


def cmd_run(args: argparse.Namespace) -> int:
    workflow = _workflow(args)
    documents = _load_documents(Path(args.path))
    if not documents:
        print("no documents found", file=sys.stderr)
        return EXIT_USAGE

    states = [workflow.run(document) for document in documents]
    if args.json:
        print(json.dumps([s.to_dict() for s in states], indent=2))
    else:
        for state in states:
            print(_render(state, show_summary=not args.no_summary))
            print()
    return _exit_code(states)


def cmd_batch(args: argparse.Namespace) -> int:
    workflow = _workflow(args)
    documents = _load_documents(Path(args.path))
    if not documents:
        print("no documents found", file=sys.stderr)
        return EXIT_USAGE

    states = workflow.run_batch(documents)
    summary = summarize_batch(states)

    if args.json:
        print(json.dumps({"summary": summary, "escalations": escalations(states)}, indent=2))
    else:
        print(f"documents           : {summary['documents']}")
        for decision, count in sorted(summary["decisions"].items()):
            print(f"  {decision:<18} {count}")
        print(f"straight-through    : {summary['straight_through_rate']:.1%}")
        print(f"needs review        : {summary['review_rate']:.1%}")
        print(f"mean confidence     : {summary['mean_confidence']:.2f}")
        print(f"mean duration       : {summary['mean_duration_ms']:.1f} ms")
        if summary["top_issues"]:
            print("\nmost frequent issues:")
            for code, count in summary["top_issues"]:
                print(f"  {code:<16} {count}")
        queue = escalations(states)
        if queue:
            print(f"\nreview queue ({len(queue)}), worst first:")
            for item in queue:
                print(
                    f"  {item['doc_id']:<12} {item['decision']:<13} "
                    f"{item['max_severity']:<9} {item['reasons'][0] if item['reasons'] else ''}"
                )

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(s.to_dict()) for s in states) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {len(states)} records to {args.out}")

    return _exit_code(states)


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the bundled samples and check each against its expected outcome."""
    path = DATA_DIR / "sample_documents.json"
    if not path.exists():
        print(f"sample data not found at {path}", file=sys.stderr)
        return EXIT_USAGE

    payload = json.loads(path.read_text(encoding="utf-8"))
    workflow = _workflow(args)

    mismatches = 0
    states = []
    print(f"{'DOC':<9}{'EXPECTED':<15}{'ACTUAL':<15}{'FIELDS':<8}{'CONF':<7}LABEL")
    for item in payload:
        state = workflow.run(Document(text=item["text"], doc_id=item["doc_id"]))
        states.append(state)
        expected = item["expected_decision"]
        actual = state.decision.value
        if expected != actual:
            mismatches += 1
        flag = " " if expected == actual else "!"
        print(
            f"{flag}{item['doc_id']:<8}{expected:<15}{actual:<15}"
            f"{len(state.fields):<8}{state.mean_confidence:<7.2f}{item['label']}"
        )

    summary = summarize_batch(states)
    print(
        f"\nstraight-through {summary['straight_through_rate']:.0%}  "
        f"review {summary['review_rate']:.0%}  "
        f"mean {summary['mean_duration_ms']:.1f} ms/document"
    )
    if mismatches:
        print(f"\n{mismatches} document(s) did not match the expected decision", file=sys.stderr)
        return EXIT_REVIEW
    return EXIT_OK


def cmd_review(args: argparse.Namespace) -> int:
    """Print the human review queue from a saved batch."""
    records = [
        json.loads(line)
        for line in Path(args.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queue = [r for r in records if r["decision"] != "complete"]
    queue.sort(key=lambda r: (r["max_severity"] != "critical", r["mean_confidence"]))

    if args.json:
        print(json.dumps(queue, indent=2))
        return EXIT_OK if not queue else EXIT_REVIEW

    if not queue:
        print("review queue is empty")
        return EXIT_OK

    print(f"{len(queue)} document(s) awaiting review\n")
    for record in queue:
        print(f"{record['doc_id']}  [{record['decision']}]  severity={record['max_severity']}  "
              f"confidence={record['mean_confidence']:.2f}")
        for reason in record["review_reasons"]:
            print(f"    - {reason}")
        print()
    return EXIT_REVIEW


def cmd_redact(args: argparse.Namespace) -> int:
    """Show exactly what a model provider would receive."""
    text = Path(args.path).read_text(encoding="utf-8") if Path(args.path).exists() else args.path
    result = redact_phi(text)
    issues = screen_document(text)

    if args.json:
        print(json.dumps({
            "entities": result.entities,
            "redacted": result.text,
            "issues": [i.to_dict() for i in issues],
        }, indent=2))
        return EXIT_OK

    print(f"PHI entities removed: {', '.join(result.entities) or 'none'}")
    if issues:
        print("\ninjection screening:")
        for issue in issues:
            print(f"  {issue.code} {issue.severity.value} {issue.message}")
    print("\n--- text as the model would see it ---")
    print(result.text)
    return EXIT_REJECTED if any(i.blocking for i in issues) else EXIT_OK


def cmd_kb(args: argparse.Namespace) -> int:
    knowledge_base = (
        KnowledgeBase.from_file(args.knowledge_base)
        if args.knowledge_base
        else KnowledgeBase.default()
    )
    if len(knowledge_base) == 0:
        print("knowledge base is empty", file=sys.stderr)
        return EXIT_USAGE

    hits = knowledge_base.search(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], indent=2))
        return EXIT_OK

    if not hits:
        print("no matching policy passages")
        return EXIT_OK
    for hit in hits:
        print(f"[{hit.score:.2f}] {hit.passage.citation}  {hit.passage.title}")
        print(f"        matched: {', '.join(hit.matched_terms[:6])}")
        print(f"        {hit.passage.text[:220]}...")
        print()
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clinicflow",
        description="Multi-agent clinical document workflows with responsible-AI controls.",
    )
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", default="null",
                       choices=["null", "echo", "ollama", "openai-compat", "vllm"],
                       help="model provider; the default runs rules only")
        p.add_argument("--model", default="llama3.2")
        p.add_argument("--knowledge-base", help="path to a JSON policy corpus")
        p.add_argument("--audit-log", help="append run records to this JSONL file")
        p.add_argument("--no-redact", action="store_true",
                       help="skip PHI redaction (never do this outside a local test)")
        p.add_argument("--min-confidence", type=float,
                       help="field confidence below which a document is escalated")
        p.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="process one document or a directory")
    run.add_argument("path")
    run.add_argument("--no-summary", action="store_true")
    add_common(run)
    run.set_defaults(func=cmd_run)

    batch = sub.add_parser("batch", help="process many documents and report aggregates")
    batch.add_argument("path")
    batch.add_argument("--out", help="write one JSON record per line to this file")
    add_common(batch)
    batch.set_defaults(func=cmd_batch)

    demo = sub.add_parser("demo", help="run the bundled samples against expected outcomes")
    add_common(demo)
    demo.set_defaults(func=cmd_demo)

    review = sub.add_parser("review", help="show the human review queue from a batch file")
    review.add_argument("path")
    review.add_argument("--json", action="store_true")
    review.set_defaults(func=cmd_review)

    redact = sub.add_parser("redact", help="show what a model provider would receive")
    redact.add_argument("path", help="a file path, or literal text")
    redact.add_argument("--json", action="store_true")
    redact.set_defaults(func=cmd_redact)

    kb = sub.add_parser("kb", help="query the policy knowledge base")
    kb.add_argument("query")
    kb.add_argument("--knowledge-base")
    kb.add_argument("--top-k", type=int, default=4)
    kb.add_argument("--json", action="store_true")
    kb.set_defaults(func=cmd_kb)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"clinicflow {__version__}")
        return EXIT_OK

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE

    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
