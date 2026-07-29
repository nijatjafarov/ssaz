import time

from ssaz import (
    AzSearchEngine,
    EvaluationHarness,
    get_chunker,
    get_embedder,
    load_documents,
    load_queries,
    show_results,
)

embedder = get_embedder("openai")

chunker = get_chunker("two-pass")

engine = AzSearchEngine(
    embedder=embedder,
    chunker=chunker,
    backend="pinecone",
    backend_options={
        "index_name": f"ssaz-azragbench",
        "namespace": "azragbench",
        "metric": "cosine",
        "cloud": "aws",
        "region": "us-east-1",
        "dimension": embedder.dim,
    },
)

# Load the corpus
documents = load_documents("examples/dataset.json")[:500]
print(f"Loaded {len(documents)} documents.")


started = time.perf_counter()
n_chunks = engine.add_documents(documents, progress=True)
print(f"Indexed {n_chunks} chunks in "
        f"{(time.perf_counter() - started) / 60:.1f} min.")

engine.export_chunks("examples/exported_chunks.json")


# Load the golden questions
queries = load_queries("examples/golden_dataset.json", documents=documents)
print(f"{len(queries)} evaluable golden questions.")

# Semantic search: show the top-k results for a few questions
for question in queries[:2]:
    show_results(question.query, engine.search(question.query, k=5),
                 question.relevant_ids)

# Evaluate every golden question through the same search path
report = EvaluationHarness(engine).evaluate(queries, k=5)
print()
print(report.to_markdown())


# User's query
user_query = "Azərbaycan Respublikasının Süni İntellekt strategiyası"
show_results(user_query, engine.search(user_query, k=5))
