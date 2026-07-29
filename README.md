# SSAZ: Semantic Search for Azerbaijani

SSAZ is an open-source Python library for building semantic search over
Azerbaijani text. You give it documents, it splits them, turns them into
vectors, stores them, and finds the right passage when someone asks a
question in plain Azerbaijani.

Every step already knows the language. You do not have to fix the casing
rules, the old Cyrillic spelling, or the legal document structure yourself.

SSAZ handles all of this by default, so you can focus on your application.

---

## Installation

```bash
pip install ssaz
```

That is enough to start. The base install is small and includes everything
the default setup needs: `requests` for the API embedding models and the
`pinecone` client for the vector store.

Optional extras, installed only when you need them:

| Extra | Command | What it adds | Size |
|---|---|---|---|
| `dense` | `pip install ssaz[dense]` | Local models BGE-M3 and Arctic (runs on your own machine) | large, ~2 GB |
| `chroma` | `pip install ssaz[chroma]` | ChromaDB as a local vector store | medium |
| `all` | `pip install ssaz[all]` | both of the above | large |

If you choose a component whose extra is missing, SSAZ tells you exactly
which command to run. The package name on PyPI is `ssaz`, and you import it
as `import ssaz`.

---

## Quickstart

Then index a document and search it:

```python
from ssaz import AzSearchEngine, Document

engine = AzSearchEngine()

engine.add_documents([
    Document(doc_id="law-13", domain="legal", text=(
        "Maddə 13. Mülkiyyət\n"
        "Azərbaycan Respublikasında mülkiyyət toxunulmazdır və dövlət "
        "tərəfindən müdafiə olunur.")),
])

for result in engine.search("mülkiyyət hüququ", k=5):
    print(result.rank, result.doc_id, result.score)
```

`AzSearchEngine()` with no arguments uses OpenAI embeddings and Pinecone.
The Pinecone index is created for you, with the right vector size.

Want to run without any API keys? Change one line:

```python
engine = AzSearchEngine(embedder="bge-m3", backend="memory")
```

---

## Architecture

![SSAZ architecture](docs/architecture.svg)

The diagram follows one article of the Constitution and one question all the
way through. Text is normalised, cut on its own markers, tagged, embedded, and
stored as a point among other points. The question then travels the same road —
the same normalizer, the same embedder — and comes back as the nearest
passages. Every box can be swapped by name.

---

## What each stage does

| Stage | Default | What it does for Azerbaijani |
|---|---|---|
| Normalizer | `AzNormalizer` | Correct `İ/ı` casing, repairs old schwa spellings, converts Cyrillic to Latin |
| Chunker | `TwoPassChunker` | Splits on real document markers (`Maddə 12.`, news datelines, wiki headings), then falls back to character splitting |
| Enrichment | `MetadataEnricher` | Adds domain, section heading, and date to every chunk |
| Embedder | `openai` | Any model by name: `gemini`, `bge-m3`, `arctic`, `qwen3`, `hf-api` |
| Index | `pinecone` | Also `memory` (no setup) and `chroma` (local file) |
| Retrieval | dense | Cosine similarity, with optional metadata filters |

Every stage can be replaced. Nothing is hardcoded.

---

## Choosing an embedding model

The embedding model decides how well search works, so we measured four of
them on [**AzRAGBench**](https://github.com/nijatjafarov/AzRAGBench), a benchmark of Azerbaijani legal, news, and
encyclopedic documents:

| Embedding model | MRR | Recall@5 | nDCG@5 |
|---|---|---|---|
| **Gemini Embedding 2** | **0.78** | **0.68** | **0.66** |
| BGE-M3 | 0.74 | 0.66 | 0.63 |
| Qwen3-Embedding | 0.74 | 0.64 | 0.62 |
| Snowflake Arctic | 0.73 | 0.65 | 0.61 |

Switching is one line:

```python
engine = AzSearchEngine(embedder="gemini")     # highest scores
engine = AzSearchEngine(embedder="bge-m3")     # local, no API key
```

Two practical notes. Vectors from different models are not comparable, so
use a separate Pinecone index per model. And you can always
re-run the benchmark yourself on your own data, see measuring quality
below.

---

## Loading your own data

You do not need to reshape your files. The loaders read JSON, JSONL, plain
`.txt`, or a folder of `.txt` files, and they recognise common field names
automatically (`id` or `doc_id`, `text` or `content`, `published_at` or
`date`).

```python
from ssaz import load_documents, load_queries

documents = load_documents("dataset.json")
queries = load_queries("golden_dataset.json", documents=documents)
```

If your field names are different, map them:

```python
documents = load_documents("court_decisions.json",
                           field_map={"doc_id": "article_no",
                                      "text": "body",
                                      "date": "issued_on"})
```
---

## Showing results

```python
from ssaz import show_results

show_results(question, engine.search(question, k=5))
```

The first result is marked as the answer, and each line shows the document
id, the chunk id, and the matching text. To change the layout, use
`ResultPresenter(style="compact")` or `style="markdown"`.

---

## Exporting

```python
from ssaz import export_data

engine.export_chunks("chunks.jsonl")      # every indexed chunk
export_data(results, "results.csv")       # search results, opens in Excel
export_data(report, "report.md")          # evaluation report
```

The format comes from the file extension: `.json`, `.jsonl`, `.csv`, `.md`.

---

## Measuring quality

The evaluation harness runs your questions through the same search code
real users hit, so the numbers are honest. It reports MRR, Recall@k, and
nDCG@k — the same metrics as the table above, and always breaks them down
by domain.

```python
from ssaz import EvaluationHarness

report = EvaluationHarness(engine).evaluate(queries, k=5)
print(report.to_markdown())
```

To compare two models, index each into its own Pinecone index and run the
harness twice.

---

## Command line

The same pipeline works from your terminal:

```bash
ssaz index corpus.jsonl --domain legal --index-name my-corpus
ssaz search "mülkiyyət hüququ nədir?" --index-name my-corpus -k 5
ssaz eval queries.json --index-name my-corpus --out results.json
ssaz chunk document.txt --domain news
```

`ssaz chunk` works offline and prints how a document would be split, useful
for checking the chunker before you spend money on embeddings.

---

## Extending the library

SSAZ is built to be extended. Register your component under a name, and it
works everywhere a built-in name works.

```python
from ssaz import register_embedder, AzSearchEngine

@register_embedder("labse")
def build_labse(**kwargs):
    from ssaz.embeddings.sentence_transformer import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder("sentence-transformers/LaBSE", **kwargs)

engine = AzSearchEngine(embedder="labse")
```

The same pattern applies to the other parts:

| You want to add | Use |
|---|---|
| A new embedding model | `@register_embedder("name")` |
| A new vector store (Qdrant, FAISS, ...) | `@register_index_backend("name")` |
| A new chunking strategy | `@register_chunker("name")` |
| A new input file type (CSV, XML, ...) | `@register_corpus_format(".ext")` |
| A new export format | `@register_export_format(".ext")` |

You can also teach the existing chunker about new document types without
writing a chunker at all:

```python
def qerar_rule(line, domain):
    m = re.match(r"^\s*QƏRAR\s+№\s*(\S+)", line)
    if m:
        return "decision", line.strip(), {"decision_no": m.group(1)}
    return None

engine = AzSearchEngine(chunker=TwoPassChunker(structural_rules=[qerar_rule]))
```

Or add your own cleanup step to the normalizer:

```python
engine = AzSearchEngine(normalizer=AzNormalizer(extra_steps=[fix_ocr_artifacts]))
```

---

## Contributing

**Azerbaijani NLP needs more hands, and this library is a good place to
start.** It is built so that a useful contribution can be small and
self-contained, you rarely need to touch the core.

Ideas that would help a lot:

- **More structural rules.** Court decisions, medical protocols, school
  textbooks, religious texts — each has its own markers. A rule is about
  ten lines of code.
- **More embedding models.** Add a model, run the benchmark, and send us the
  numbers. The table above should keep growing.
- **More vector stores.** Qdrant, Milvus, FAISS, and Weaviate all fit the
  existing `VectorIndex` interface.
- **Better morphology.** A proper Azerbaijani stemmer or morphological
  analyser would improve retrieval further.
- **More test data.** Real Azerbaijani documents and question sets from new
  domains make the benchmark stronger for everyone.
- **Documentation and examples**, including in Azerbaijani.

How to contribute:

1. Fork the repository and create a branch.
2. Add your change together with a test.
3. Run `pytest` and make sure everything passes.
4. Open a pull request describing what you added and why.
---

## Development

```bash
pip install -e .[dev]
pytest
```

---

## Citation

SSAZ is part of the master's thesis *"Development of NLP Services and Tools
for the Azerbaijani Language"* (N. Jafarov, ADA University & George Washington University).

## License

MIT - free to use, change, and share.
