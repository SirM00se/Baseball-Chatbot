import os
import sys
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rag_engine import RAGEngine

DATA_PATH = BASE_DIR / "data" / "eval_dataset.json"
OUTPUT_PATH = BASE_DIR / "ragas_results.csv"


def run_ragas():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please export OPENAI_API_KEY='your-key' before running this script."
        )

    with DATA_PATH.open(encoding="utf-8") as file:
        test_cases = json.load(file)

    if not test_cases:
        raise ValueError(f"No evaluation cases found: {DATA_PATH}")

    eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    engine = RAGEngine()
    samples = []

    for i, item in enumerate(test_cases):
        question = item["question"]
        session = engine.get_session(f"eval_{i}")

        # Retrieve documents
        embedding = engine.create_embedding_question(question)
        ids, scores = engine.generate_ids(embedding, k=5)
        chunks = engine.query_sqlite(ids, scores) if ids else []

        contexts = [
            chunk["text"]
            for chunk in chunks
            if chunk.get("text")
        ]

        # Generate answer
        prompt = engine.make_json_prompt(question, chunks, session)
        answer = engine.call_llm(prompt)

        samples.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item["ground_truth_answer"],
        })

        print(f"Prepared {i + 1}/{len(test_cases)} cases")

    dataset = EvaluationDataset.from_list(samples)

    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        embeddings=eval_embeddings,
        run_config=RunConfig(max_workers=4, timeout=300),
    )

    print("\nEvaluation results:")
    print(result)

    result.to_pandas().to_csv(OUTPUT_PATH, index=False)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_ragas()