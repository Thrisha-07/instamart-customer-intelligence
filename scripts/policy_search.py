
from pathlib import Path
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm

POLICY_DIR = ROOT / "data" / "policies"
EVAL_PATH = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
TOP_K = 3


def load_policy_chunks(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split policy files into word chunks with configurable overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    chunks = []
    stride = chunk_size - overlap

    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()

        for start in range(0, len(words), stride):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                break

            chunks.append({
                "source": path.name,
                "text": " ".join(chunk_words),
            })

            if start + chunk_size >= len(words):
                break

    if not chunks:
        raise ValueError(f"No policy text found in {POLICY_DIR}")

    return chunks


class PolicySearch:
    def __init__(self, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.chunks = load_policy_chunks(chunk_size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self, query, k=TOP_K):
        """Return the top-k chunks by cosine similarity."""
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector, self.document_vectors
        ).ravel()

        ranked_indices = scores.argsort()[::-1][:k]

        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
                "text": self.chunks[i]["text"],
            }
            for i in ranked_indices
        ]

    def answer(self, question):
        """Answer using retrieved policy context only."""
        results = self.retrieve(question, k=TOP_K)

        context_parts = []
        for item in results:
            context_parts.append(
                f"[{item['source']}]\n{item['text']}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = """
POLICY ASSISTANT

Answer the user's question using only the supplied policy context.
Do not invent policy details or use outside knowledge.
If the supplied context does not contain the answer, say:
I could not find that information in the policy documents.
"""

        user_message = (
            f"CONTEXT:\n{context}\n"
            f"QUESTION:\n{question}"
        )

        response = mock_llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ])

        return {
            "answer": response.choices[0].message.content,
            "sources": results,
        }


def evaluate_search(overlap, output_name):
    """Evaluate top-3 document retrieval and policy answers."""
    search = PolicySearch(chunk_size=40, overlap=overlap)
    evaluation = pd.read_csv(EVAL_PATH, keep_default_na=False)

    rows = []

    for _, row in evaluation.iterrows():
        question = str(row["question"]).strip()
        expected_doc = Path(
            str(row["expected_doc"]).strip()
        ).name
        expected_fact = str(row["expected_fact"]).strip()

        retrieved = search.retrieve(question, k=TOP_K)
        answer_result = search.answer(question)
        answer = answer_result["answer"]

        if expected_doc.casefold() in {"", "none"}:
            doc_match = None
            fact_match = None
            fallback_match = answer.startswith(
                "I could not find"
            )
            status = "OUT_OF_SCOPE"
        else:
            doc_match = any(
                item["source"].casefold() == expected_doc.casefold()
                for item in retrieved
            )
            fact_match = expected_fact in answer
            fallback_match = None
            status = "ANSWERABLE"

        rows.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_docs": ";".join(
                item["source"] for item in retrieved
            ),
            "expected_fact": expected_fact,
            "answer": answer,
            "document_match_top3": doc_match,
            "expected_fact_in_answer": fact_match,
            "fallback_correct": fallback_match,
            "status": status,
        })

    results = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / output_name
    results.to_csv(output_path, index=False)

    answerable = results[results["status"] == "ANSWERABLE"]
    out_of_scope = results[results["status"] == "OUT_OF_SCOPE"]

    print(f"\nEvaluation with overlap={overlap}")
    print("Chunks:", len(search.chunks))
    print("Questions:", len(results))

    if not answerable.empty:
        print(
            "Expected document in top 3:",
            f"{answerable['document_match_top3'].mean():.1%}",
        )
        print(
            "Expected fact in answer:",
            f"{answerable['expected_fact_in_answer'].mean():.1%}",
        )

    if not out_of_scope.empty:
        print(
            "Correct out-of-scope fallback:",
            f"{out_of_scope['fallback_correct'].mean():.1%}",
        )

    print("Saved:", output_path)
    return results, search


def main():
    mock_llm.reset()

    # Main retrieval evaluation: 40-word chunks, 10-word overlap.
    overlap_results, search = evaluate_search(
        overlap=10,
        output_name="policy_search_results.csv",
    )

    print("\nExample policy answer:")
    example = search.answer(
        "When can a customer receive a full refund for late delivery?"
    )
    print(example["answer"])
    print("Sources:", [item["source"] for item in example["sources"]])

    # Compare against chunks without overlap.
    mock_llm.reset()
    no_overlap_results, _ = evaluate_search(
        overlap=0,
        output_name="policy_search_no_overlap_results.csv",
    )

    print("\nOverlap comparison:")
    print(
        "With 10-word overlap, expected-fact rate:",
        overlap_results[
            overlap_results["status"] == "ANSWERABLE"
        ]["expected_fact_in_answer"].mean(),
    )
    print(
        "With no overlap, expected-fact rate:",
        no_overlap_results[
            no_overlap_results["status"] == "ANSWERABLE"
        ]["expected_fact_in_answer"].mean(),
    )


if __name__ == "__main__":
    main()
