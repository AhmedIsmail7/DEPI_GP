import argparse

from modules.retrieval import get_retriever
from modules.llm_handler import get_llm_handler


def answer_query(query: str, top_k: int = 3):
    retriever = get_retriever()
    context_chunks = retriever.retrieve(query, top_k=top_k)
    handler = get_llm_handler()
    answer = handler.generate_response(query, context_chunks)
    return answer, context_chunks


def main():
    parser = argparse.ArgumentParser(description="Ask a question about the stored video content")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=3, help="Number of context chunks to retrieve")
    args = parser.parse_args()

    answer, context_chunks = answer_query(args.query, top_k=args.top_k)
    print(f"\nQuery: {args.query}\n")
    print("Retrieved context:")
    for chunk in context_chunks:
        print(f"- [{chunk['timestamp']}s] {chunk['text']}")
    print("\nCohere answer:")
    print(answer)


if __name__ == "__main__":
    main()
