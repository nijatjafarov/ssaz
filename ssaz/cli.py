"""
SAZ command-line interface.

Usage:

    ssaz index corpus_dir/ --domain legal --save index.json
    ssaz search "mülkiyyət hüququ nədir?" --load index.json -k 5
    ssaz eval queries.json --load index.json --out results.json
    ssaz chunk document.txt --domain news
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from ssaz import __version__
from ssaz.chunking import TwoPassChunker
from ssaz.data import load_documents
from ssaz.documents import Document
from ssaz.engine import AzSearchEngine
from ssaz.evaluation import EvaluationHarness
from ssaz.text import AzNormalizer


def _build_engine(args: argparse.Namespace) -> AzSearchEngine:
    backend_options = {}
    if args.backend == "chroma":
        backend_options["collection_name"] = args.collection
        if args.persist_dir:
            backend_options["persist_directory"] = args.persist_dir
    elif args.backend == "pinecone":
        backend_options["index_name"] = args.index_name
        if args.namespace:
            backend_options["namespace"] = args.namespace
        if args.dimension:
            backend_options["dimension"] = args.dimension
    return AzSearchEngine(embedder=args.embedder, backend=args.backend,
                          backend_options=backend_options)


def cmd_index(args: argparse.Namespace) -> int:
    engine = _build_engine(args)
    documents = load_documents(args.path, default_domain=args.domain)
    if not documents:
        print("No documents found.", file=sys.stderr)
        return 1
    n_chunks = engine.add_documents(documents)
    print(f"Indexed {len(documents)} documents -> {n_chunks} chunks "
          f"(embedder={engine.embedder.name}).")
    if args.save:
        engine.save(args.save)
        print(f"Index saved to {args.save}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    engine = _build_engine(args)
    if args.load:
        engine.load(args.load)
    if engine.count() == 0:
        print("Index is empty — run 'ssaz index' first.", file=sys.stderr)
        return 1
    where = {"domain": args.filter_domain} if args.filter_domain else None
    results = engine.search(args.query, k=args.k, where=where)
    if args.json:
        print(json.dumps([r.__dict__ for r in results], ensure_ascii=False,
                         indent=2))
        return 0
    for result in results:
        heading = result.metadata.get("section_heading", "")
        print(f"\n#{result.rank}  score={result.score:.4f}  "
              f"[{result.metadata.get('domain', '?')}] "
              f"{result.doc_id}" + (f" — {heading}" if heading else ""))
        text = result.text if len(result.text) <= 400 else result.text[:400] + "…"
        print(f"   {text}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    engine = _build_engine(args)
    if args.load:
        engine.load(args.load)
    if engine.count() == 0:
        print("Index is empty — run 'ssaz index' first.", file=sys.stderr)
        return 1
    harness = EvaluationHarness(engine)
    queries = harness.load_queries(args.queries)
    report = harness.evaluate(queries, k=args.k)
    print(report.to_markdown())
    if args.out:
        Path(args.out).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\nResults written to {args.out}")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    """Inspect chunker output for a single document (used for the manual
    chunk-inspection step in the risk-mitigation plan)"""
    normalizer = AzNormalizer()
    chunker = TwoPassChunker(chunk_size=args.chunk_size)
    text = normalizer(Path(args.path).read_text(encoding="utf-8"))
    document = Document(doc_id=Path(args.path).stem, text=text,
                        domain=args.domain)
    for chunk in chunker.chunk(document):
        heading = chunk.metadata.get("section_heading", "-")
        print(f"--- {chunk.chunk_id}  [{chunk.metadata.get('chunking')}] "
              f"{heading}  ({len(chunk)} chars)")
        print(chunk.text[:300] + ("…" if len(chunk.text) > 300 else ""))
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ssaz",
        description="SAZ — semantic search for the Azerbaijani language.")
    parser.add_argument("--version", action="version",
                        version=f"ssaz {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_engine_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--embedder", default="openai",
                       help="openai | bge-m3 | arctic | hf-api | qwen3 | gemini")
        p.add_argument("--backend", default="pinecone",
                       choices=["pinecone", "memory", "chroma"])
        p.add_argument("--collection", default="ssaz",
                       help="chroma collection name")
        p.add_argument("--persist-dir", default=None,
                       help="chroma persistence directory")
        p.add_argument("--index-name", default="ssaz",
                       help="pinecone index name")
        p.add_argument("--namespace", default="",
                       help="pinecone namespace")
        p.add_argument("--dimension", type=int, default=None,
                       help="embedding dimension (required to create a "
                            "new pinecone index)")

    p_index = subparsers.add_parser("index", help="index a corpus")
    p_index.add_argument("path", help=".txt file, .jsonl file, or directory")
    p_index.add_argument("--domain", default="general",
                         help="legal | news | encyclopedic | general")
    p_index.add_argument("--save", default=None,
                         help="save memory index to JSON")
    add_engine_args(p_index)
    p_index.set_defaults(func=cmd_index)

    p_search = subparsers.add_parser("search", help="query an index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--filter-domain", default=None)
    p_search.add_argument("--load", default=None,
                          help="load memory index from JSON")
    p_search.add_argument("--json", action="store_true",
                          help="print results as JSON")
    add_engine_args(p_search)
    p_search.set_defaults(func=cmd_search)

    p_eval = subparsers.add_parser("eval", help="run the evaluation harness")
    p_eval.add_argument("queries", help="JSON file of evaluation queries")
    p_eval.add_argument("-k", type=int, default=5)
    p_eval.add_argument("--load", default=None)
    p_eval.add_argument("--out", default=None,
                        help="write JSON results to this path")
    add_engine_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_chunk = subparsers.add_parser("chunk", help="inspect chunker output")
    p_chunk.add_argument("path")
    p_chunk.add_argument("--domain", default="general")
    p_chunk.add_argument("--chunk-size", type=int, default=800)
    p_chunk.set_defaults(func=cmd_chunk)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
