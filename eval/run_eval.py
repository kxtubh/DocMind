"""
Retrieval + generation evaluation.

Usage:
  # 1. Ingest sample corpus first (run once):
  python run_eval.py --ingest

  # 2. Run retrieval eval (recall@k, MRR, hit rate):
  python run_eval.py --retrieval

  # 3. Run RAGAS generation eval (faithfulness, relevancy, precision):
  python run_eval.py --ragas

  # 4. Compare chunking strategies (ingest with 3 strategies, eval each):
  python run_eval.py --compare

All flags can be combined: python run_eval.py --ingest --retrieval --ragas
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv("../.env.local")

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
SAMPLES_DIR = Path("../data/samples")
DATASET_PATH = Path("dataset.json")

CHUNKING_STRATEGIES = [
    {"strategy": "recursive", "chunk_size": 500,  "chunk_overlap": 50,  "namespace": "recursive-500"},
    {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 200, "namespace": "recursive-1000"},
    {"strategy": "fixed",     "chunk_size": 500,  "chunk_overlap": 0,   "namespace": "fixed-500"},
]


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_corpus(namespace: str = "", strategy: str = "recursive",
                  chunk_size: int = 1000, chunk_overlap: int = 200):
    """Ingest all sample markdown files into Pinecone."""
    for md_file in SAMPLES_DIR.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        payload = {
            "text": text,
            "source": md_file.name,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "namespace": namespace,
        }
        r = requests.post(f"{API_URL}/ingest", json=payload)
        r.raise_for_status()
        data = r.json()
        print(f"  ingested {md_file.name}: {data['chunks']} chunks → namespace '{namespace or 'default'}'")


# ── Retrieval eval ────────────────────────────────────────────────────────────

def recall_at_k(retrieved_sources: list[str], gold_source: str, k: int) -> float:
    return 1.0 if gold_source in retrieved_sources[:k] else 0.0


def mrr(retrieved_sources: list[str], gold_source: str) -> float:
    for rank, src in enumerate(retrieved_sources, start=1):
        if src == gold_source:
            return 1.0 / rank
    return 0.0


def run_retrieval_eval(namespace: str = "", k: int = 5, label: str = "default") -> dict:
    dataset = load_dataset()
    r_at_1, r_at_3, r_at_5, mrr_scores, hits = [], [], [], [], []

    for item in dataset:
        r = requests.post(f"{API_URL}/query", json={"question": item["question"], "namespace": namespace, "k": k})
        r.raise_for_status()
        data = r.json()
        sources = [s["metadata"].get("source", "") for s in data["sources"]]

        r_at_1.append(recall_at_k(sources, item["gold_source"], 1))
        r_at_3.append(recall_at_k(sources, item["gold_source"], 3))
        r_at_5.append(recall_at_k(sources, item["gold_source"], 5))
        mrr_scores.append(mrr(sources, item["gold_source"]))
        hits.append(1.0 if item["gold_source"] in sources else 0.0)

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    result = {
        "label": label,
        "n": len(dataset),
        "recall@1": avg(r_at_1),
        "recall@3": avg(r_at_3),
        "recall@5": avg(r_at_5),
        "mrr": avg(mrr_scores),
        "hit_rate": avg(hits),
    }
    return result


# ── RAGAS eval ────────────────────────────────────────────────────────────────

def run_ragas_eval(namespace: str = "") -> dict:
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
    except ImportError:
        print("  RAGAS not installed. Run: pip install ragas datasets")
        return {}

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    dataset = load_dataset()
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in dataset:
        r = requests.post(f"{API_URL}/query", json={"question": item["question"], "namespace": namespace, "k": 5})
        r.raise_for_status()
        data = r.json()
        rows["question"].append(item["question"])
        rows["answer"].append(data["answer"])
        rows["contexts"].append([s["pageContent"] for s in data["sources"]])
        rows["ground_truth"].append(item["gold_answer"])

    ds = Dataset.from_dict(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ChatOpenAI(model="gpt-4o-mini"),
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    return {
        "faithfulness": round(float(result["faithfulness"]), 3),
        "answer_relevancy": round(float(result["answer_relevancy"]), 3),
        "context_precision": round(float(result["context_precision"]), 3),
    }


# ── Comparison across strategies ──────────────────────────────────────────────

def run_comparison():
    print("\n=== Chunking strategy comparison ===\n")
    print("Ingesting corpus with 3 strategies…")
    for cfg in CHUNKING_STRATEGIES:
        print(f"\n  [{cfg['namespace']}]")
        ingest_corpus(
            namespace=cfg["namespace"],
            strategy=cfg["strategy"],
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )

    print("\nRunning retrieval eval per strategy…\n")
    results = []
    for cfg in CHUNKING_STRATEGIES:
        r = run_retrieval_eval(namespace=cfg["namespace"], k=5, label=cfg["namespace"])
        results.append(r)
        print(f"  {r['label']:20s}  recall@5={r['recall@5']}  mrr={r['mrr']}  hit_rate={r['hit_rate']}")

    print("\n=== Results table ===")
    print(f"{'Strategy':<22} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'Hit%':>6}")
    print("-" * 54)
    for r in results:
        print(f"{r['label']:<22} {r['recall@1']:>6} {r['recall@3']:>6} {r['recall@5']:>6} {r['mrr']:>6} {r['hit_rate']:>6}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--ingest",    action="store_true", help="Ingest sample corpus (default namespace)")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval eval (recall@k, MRR, hit rate)")
    parser.add_argument("--ragas",     action="store_true", help="Run RAGAS generation eval")
    parser.add_argument("--compare",   action="store_true", help="Compare 3 chunking strategies end-to-end")
    parser.add_argument("--namespace", default="", help="Pinecone namespace to query (default: '')")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    # Confirm API is up
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"API healthy: {r.json()}\n")
    except Exception as e:
        print(f"Cannot reach API at {API_URL}: {e}")
        sys.exit(1)

    if args.compare:
        run_comparison()
        return

    if args.ingest:
        print("Ingesting sample corpus…")
        ingest_corpus(namespace=args.namespace)
        print("Done.\n")

    if args.retrieval:
        print("Running retrieval eval…")
        result = run_retrieval_eval(namespace=args.namespace, k=5)
        print(json.dumps(result, indent=2))

    if args.ragas:
        print("\nRunning RAGAS eval…")
        result = run_ragas_eval(namespace=args.namespace)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
