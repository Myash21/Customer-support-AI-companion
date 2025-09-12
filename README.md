## Customer Support AI Companion

A production-minded prototype for end-to-end ticket understanding and assistance:

- **Ticket Classification**: Topic tags, sentiment, and priority from raw emails or tickets
- **Hybrid RAG**: Dense + BM25 retrieval over product docs and developer guides
- **Domain expert notification**: For non-RAG topics, enter an expert's email and send a notification with ticket details
- **Classification Stats**: Bulk analytics and charts over sample tickets

Runs as a lightweight app (see `app.py`) with a focus on practical accuracy, latency, and cost-awareness.

## Quick Start

1) Environment

- Python 3.9+
- Set these environment variables before running:

```bash
GOOGLE_API_KEY="<your_key>"
SMTP_HOST="smtp.gmail.com"
SMTP_PORT="587"
SMTP_USER="you@example.com"
SMTP_PASSWORD="your_app_password"
SMTP_FROM="you@example.com"
```

2) Install

```bash
pip install -r requirements.txt
```

3) Build Vector DB (scrape → chunk → embed → persist)

```bash
python utils/build_vectordb.py
```

4) Run the app

```bash
streamlit run app.py
```

## Repository Structure

- `app.py`: Streamlit app entrypoint
- `utils/build_vectordb.py`: Scrape, chunk, embed, and persist to ChromaDB
- `rag/`:
  - `rag_pipeline.py`: Main RAG pipeline and answer orchestration
  - `hybrid_retrieval_engine.py`: Dual (dense + BM25) retrieval, normalization, merge, dedupe
  - `hybrid_retrieval_adapter.py`: LangChain-compatible retriever adapter
  - `hybrid_config.py`: Central config for hybrid retrieval parameters
  - `README_hybrid.md`: Additional docs for hybrid retrieval
- `data/`: Sample tickets (`sample_tickets.json`, `all_tickets.json`)
- `requirements.txt`: Python dependencies

## 1) Ticket Classification

### Topic Categorization (subject + body)

- **Approach**: Zero-shot classification using the pre-trained model `facebook/bart-large-mnli`
- **Why zero-shot on a pre-trained model**:
  - **Data constraints**: No sizable labeled corpus for supervised fine-tuning
  - **Cost/latency**: Avoid LLM few-shot prompting for routine classification
  - **Robustness**: MNLI-trained NLI models generalize well to arbitrary label sets
- **Why `bart-large-mnli`**:
  - Trained on NLI; strong at entailment-style label matching
  - Widely adopted baseline for zero-shot classification
  - Handles both short subjects and longer bodies

Alternatives considered:

- **Supervised fine-tuning** (e.g., BERT/RoBERTa/DistilBERT) was rejected due to lack of labeled data and timeline constraints
- **Few-shot with LLMs** (GPT/Claude/Llama) was rejected due to higher cost, prompt complexity, and hallucination risk for taxonomy mapping
- **Embeddings + similarity** can struggle when label names are semantically close

### Sentiment Classification (body only)

- **Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Classes**: anger, disgust, fear, joy, neutral, sadness, surprise
- **Mapping to business-friendly labels**:
  - anger → Angry
  - disgust → Frustrated
  - fear → Frustrated
  - sadness → Frustrated
  - joy → Curious
  - surprise → Curious
  - neutral → Neutral

Rationale: Sentiment is expressed primarily in the body (tone, phrasing), not the subject. The chosen model is compact, accurate for short texts, and trained on diverse emotion datasets.

### Priority Classification (keyword rules)

- **P0 (High)**: urgent, critical, blocked, asap, immediately
- **P1 (Medium)**: soon, important, deadline, required
- **P2 (Low)**: otherwise

Rationale: Simple, deterministic, cost-free, and explainable. Emphasizes business usefulness over academic accuracy. Can be upgraded later with learned models when more labeled data exists.

## 2) Hybrid RAG System

Refer to `rag/README_hybrid.md` for an in-depth overview of the hybrid RAG architecture.

Combines dense semantic retrieval (embeddings + ChromaDB) with sparse keyword retrieval (BM25) for better coverage and precision.

### Vector DB Build Pipeline (scrape → chunk → embed)

- Single target list from concatenated `DOCS_URLS` and `DEV_URLS`
- HTTP session with retries and user-agent
- Parallel fetch with `ThreadPoolExecutor`
- HTML parsing via BeautifulSoup; extract `<title>` and page text; collapse whitespace
- Chunking via `RecursiveCharacterTextSplitter` (≈1000 chars, 150 overlap)
- SHA1 `content_hash` of `url::chunk` for dedupe
- Metadata: `source`, `title`, `chunk_index`, `content_hash`, `access_visibility`, `access_dept`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Persistence: Chroma collection `db_docs` in `CHROMA_PERSIST_DIR` (default `chroma_db`)
- Batch ingest (200 items): `add_texts(texts, metadatas, ids)` with IDs = `content_hash`

### Retrieval Engine

- **Dense**: Chroma similarity search with embeddings
- **Sparse**: BM25 keyword search
- **Combination**: Weighted scores, defaults in `rag/hybrid_config.py`:

```python
HYBRID_CONFIG = {
    "dense_weight": 0.7,
    "sparse_weight": 0.3,
    "top_k_dense": 5,
    "top_k_sparse": 5,
    "final_top_k": 3,
    "dedup_threshold": 0.8
}
```

- Normalization: Scores scaled to 0–1 for fair merging
- Deduplication: Content hashing and similarity thresholding
- Robustness: Automatic fallback to dense-only on errors; retries for transient failures

### Generation Model

- **Default**: Gemini 2.5 Flash via API for fast, high-quality responses and low memory footprint
- **Pros**: Excellent instruction following, large context, fast, cost-effective for small–medium usage
- **Cons**: External API dependency, per-query cost, privacy considerations
- **Offline alternative**: `microsoft/phi-3-mini-4k-instruct` or `meta-llama/Llama-3-8B-Instruct` for privacy/no API cost, with trade-offs in memory, latency, and response quality

### Usage in Code

```python
from rag.rag_pipeline import rag_answer

answer, sources = rag_answer("How to connect Snowflake to Atlan?")
answer, sources = rag_answer("How to connect Snowflake to Atlan?", use_hybrid=False)
```

To customize retrieval directly:

```python
from rag.hybrid_retrieval_adapter import create_hybrid_retriever
from langchain.chains import create_retrieval_chain

retriever = create_hybrid_retriever()
chain = create_retrieval_chain(retriever, document_chain)
result = chain.invoke({"input": "Your query here"})
```

## 3) Domain Expert Notification

- In the Test Individual Ticket tab, if the classified topic is not in the RAG-eligible list, the app shows:
  - A notice: "This ticket has been classified as a 'topic' issue"
  - An input to enter a domain expert's email and a button to send
- On send, the app emails the expert with ticket ID, subject, body, priority, and sentiment using `email_support/sender.py` (SMTP-based)

### Implementation details

- Trigger condition: executes for non-RAG topics (RAG topics are `How-to`, `Product`, `Best practices`, `API/SDK`, `SSO`).
- UI flow location: `app.py` → Test Individual Ticket tab → non-RAG branch.
  - The classification result is saved to `st.session_state['last_classification']` outside the `st.form()` to allow separate action buttons.
  - Email input is validated via regex before sending: `^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$`.
- Email sending function: `email_support/sender.py`
  - Signature: `send_email(recipient_email: str, subject: str, body: str) -> None`
  - Loads environment via `dotenv` if present.
  - Uses `SMTP_HOST` (default `smtp.gmail.com`) and `SMTP_PORT` (default `587`).
  - Auth required: `SMTP_USER` and `SMTP_PASSWORD` (raises if missing).
  - From address: `SMTP_FROM` (falls back to `SMTP_USER`).
  - TLS: `server.starttls()` then `server.login(...)` before `sendmail(...)`.

### Provider notes

- Gmail: create an App Password (Google Account → Security → 2‑Step Verification → App passwords) and use it as `SMTP_PASSWORD`.
- Office 365: `SMTP_HOST=smtp.office365.com`, `SMTP_PORT=587`; may require an app password or modern auth-enabled relay.


## 4) Classification Stats

- Runs classifiers over all sample tickets in bulk (`data/*.json`)
- Produces a DataFrame and charts:
  - Topic distribution bar chart
  - Sentiment distribution bar chart
  - Priority distribution bar chart
  - Quick view of top topics
- Shows a small table of sample classification results for inspection

## Configuration

- Retrieval parameters: `rag/hybrid_config.py`
- RAG preferences, logging flags, and fallbacks are exposed in config; enable additional debug info by setting:

```python
RETRIEVAL_PREFERENCES = {
    "log_retrieval_method": True,
    "show_scores": True,
}
```

## Design Decisions and Trade-offs

The following choices were explicitly optimized for deployment on Streamlit Cloud (free tier), considering limited RAM/CPU, cold starts, no GPU, and ephemeral storage.

- **Zero-shot topic classification**
  - **Decision**: Use `bart-large-mnli`
  - **Trade-off**: Lower ceiling vs fine-tuned models, but no labeling cost and avoids LLM latency/cost/hallucinations

- **Compact emotion model for sentiment**
  - **Decision**: `j-hartmann/emotion-english-distilroberta-base`
  - **Trade-off**: Emotion-to-business-label mapping is lossy but pragmatic and explainable

- **Rule-based priority**
  - **Decision**: Keyword cues for P0/P1/P2
  - **Trade-off**: Misses nuanced urgency; chosen for determinism, speed, and simplicity

- **Hybrid retrieval**
  - **Decision**: Dense (semantic) + BM25 (keyword) with weighted merge
  - **Trade-off**: ~2–3× latency vs dense-only; improved recall and precision, especially for technical terms and error codes

- **Gemini API for generation**
  - **Decision**: Use hosted model for quality and low memory usage
  - **Trade-off**: API dependency and running cost; mitigated by small payloads and caching opportunities

- **ChromaDB persistence**
  - **Decision**: On-disk persistence for reusability across runs
  - **Trade-off**: Local disk usage; enables quick startup and reproducibility

### Vector Database Choice: ChromaDB vs Alternatives

**Why ChromaDB over other vector databases:**

- **vs FAISS**: 
  - FAISS requires manual index serialization/deserialization and lacks built-in metadata filtering
  - ChromaDB provides automatic persistence, metadata queries, and easier LangChain integration
  - FAISS is more suitable for research/prototyping; ChromaDB better for production applications

- **vs Qdrant**:
  - Qdrant requires separate server deployment and network configuration
  - ChromaDB runs embedded, eliminating infrastructure complexity
  - Qdrant offers advanced features (payload filtering, geo-search) unnecessary for this use case

- **vs Pinecone**:
  - Pinecone is cloud-only with API costs and external dependencies
  - ChromaDB runs locally, ensuring data privacy and eliminating ongoing costs
  - Pinecone requires internet connectivity; ChromaDB works offline

- **vs Weaviate**:
  - Weaviate requires Docker/server setup and more complex configuration
  - ChromaDB has simpler deployment and lower resource requirements
  - Weaviate's GraphQL interface adds complexity for this straightforward use case

**Deployment Constraints Considered:**
- **Streamlit Cloud limitations**: No persistent storage, limited memory, no external services
- **Cost optimization**: Avoid cloud vector DB fees and API costs
- **Simplicity**: Minimize infrastructure dependencies and configuration complexity
- **Data privacy**: Keep sensitive documentation local without external API calls

**Why ChromaDB is included in the repo:**
- **Portability**: Complete self-contained solution that works across different environments
- **Reproducibility**: Anyone can clone and run without external service setup
- **Version control**: Vector database state can be tracked and shared
- **Offline capability**: Works without internet connectivity for retrieval
- **Development speed**: No need to set up external services during development

**Production-Grade Alternative:**
For production deployments, will be considering **Qdrant** or **Weaviate** for their advanced features:
- **Qdrant**: Better performance, horizontal scaling, advanced filtering, and production-ready clustering
- **Weaviate**: GraphQL API, multi-tenancy, enterprise security features, and better metadata management
- **Rationale**: These offer better scalability, reliability, and enterprise features compared to embedded ChromaDB

## Performance and Reliability

- Latency: Hybrid retrieval adds overhead; tune `top_k_*` and weights to balance quality vs speed
- Memory: BM25 index adds RAM; consider smaller `top_k` and chunk sizes if constrained
- Robustness: Automatic fallback to dense-only path; retry logic for transient failures
- Dedupe: Content hashing prevents index bloat and duplicate sources

## Evaluation

### Retrieval Performance

The retrieval system was manually evaluated using a set of custom support tickets to assess how well the hybrid retriever (dense + sparse) finds relevant documents. The evaluation process involved:

- **Test Dataset**: 20 custom support tickets covering various topics (permissions, connectors, SSO, API usage)
- **Ground Truth**: Manually identified relevant documents for each query from the knowledge base
- **Metric**: Recall@k (where k=3, matching the retriever's final_top_k configuration)
- **Result**: Recall@3 = 0.76

**Analysis**: The 0.76 recall score indicates that 76% of queries had at least one relevant document in the top-3 retrieved results. This performance can be improved by:
- Adding more comprehensive content to the knowledge base
- Better document chunking strategies for technical content
- Fine-tuning the hybrid retrieval weights (dense vs sparse)
- The current score was constrained by limited familiarity with the dataset during initial curation

### Generation Quality

A formal evaluation of answer quality has not been conducted yet. Future evaluation plans include:

**Heuristic Methods** (fast, deterministic):
- **Style Compliance**: Check for required sections (Context, Next steps), bullet formatting, citation presence
- **Conciseness**: Token count analysis against target ranges
- **Directness**: Count of actionable steps and imperative statements
- **Citation Format**: Verify proper source attribution structure

**LLM-as-Judge Methods** (comprehensive, requires evaluator model):
- **Answer Relevancy**: How well the response addresses the user's question intent
- **Faithfulness/Groundedness**: Degree to which claims are supported by provided context
- **Citation Accuracy**: Whether cited sources actually contain the supporting evidence
- **Hallucination Detection**: Identification of unsupported claims not in the context
- **Outside Knowledge Over-reliance**: Detection when external facts are introduced unnecessarily

**Implementation Approach**:
- Use a separate evaluator LLM (e.g., Gemini or Claude) to score generation quality
- Implement both per-sample scoring and aggregate metrics
- Establish evaluation thresholds through calibration on a small labeled subset
- Track both mean scores and failure rates for quick regression detection

## Security and Privacy

- **Data handling**: Gemini API calls send queries externally; ensure compliance with your data policies
- **PII protection**: Pre-process tickets to remove personal information before API calls
- **Access control**: Built-in metadata fields (`access_visibility`, `access_dept`) for future role-based filtering
- **Local storage**: ChromaDB keeps vector data local; no external vector database dependencies

## Future Enhancements

### Immediate Improvements
- Learned priority model when labeled data is available
- Caching layer for retrieval results and answers
- Async retrieval for improved concurrency
- Adaptive weighting based on query type

### Advanced RAG Techniques
- **Semantic chunking**: Multi-granularity chunking (paragraph + sentence) with semantic coherence
- **Domain-tuned embeddings**: Replace general-purpose embeddings with domain-specific models
- **Query optimization**: Query rewriting, paraphrasing, and acronym expansion for better hit rates
- **Re-ranking**: Add re-ranking layer after hybrid retrieval for improved precision
- **Graph RAG**: Incorporate knowledge graph structure for better reasoning and context
- **Context pruning**: Intelligent document filtering to reduce noise and improve relevance

### Production Scaling
- **Vector database upgrade**: Migrate to Qdrant or Weaviate for better performance and scalability
- **Hallucination mitigation**: Add grounding mechanisms, citation highlighting, and fact-checking subagents
- **Cost optimization**: Implement ANN (Approximate Nearest Neighbor) search and embedding dimension optimization
- **A/B testing framework**: Compare different retrieval strategies and model configurations
- **On-device models**: Configurable fallbacks for offline operation and reduced API costs

## Troubleshooting

- BM25 index build failure: Verify documents loaded and Chroma populated, and `rank-bm25` installed
- Memory issues: Reduce `top_k_dense` and `top_k_sparse`; adjust chunk size
- Performance: Enable parallel retrieval; consider caching; tune timeouts
- Debug: Set `log_retrieval_method` and `show_scores` to `True`
