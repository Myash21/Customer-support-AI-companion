# Hybrid RAG Pipeline

This implementation combines **dense retrieval** (semantic search) and **sparse retrieval** (BM25 keyword search) to provide more comprehensive and accurate document retrieval for the customer support AI companion.

## 🏗️ Architecture

```
Query → HybridRetrievalOrchestrator
    ├── Dense Search (ChromaDB + Embeddings) → Results A
    ├── Sparse Search (BM25) → Results B
    └── Combine & Re-rank → Final Results → LangChain Chain
```

## 📁 Files Overview

- **`hybrid_retrieval_engine.py`**: Core hybrid retrieval orchestrator
- **`hybrid_retrieval_adapter.py`**: LangChain retriever wrapper
- **`hybrid_config.py`**: Configuration settings
- **`rag_pipeline.py`**: Updated main pipeline with hybrid support

## 🔧 Key Features

### 1. **Dual Retrieval Methods**
- **Dense Retrieval**: Uses ChromaDB with sentence-transformers embeddings for semantic understanding
- **Sparse Retrieval**: Uses BM25 for exact keyword matching and technical term precision

### 2. **Intelligent Combination**
- **Weighted Scoring**: Prioritizes dense retrieval (70%) over sparse (30%)
- **Score Normalization**: Ensures fair comparison between different scoring methods
- **Deduplication**: Removes duplicate results based on content similarity

### 3. **Robust Fallback**
- **Automatic Fallback**: Falls back to simple dense retrieval if hybrid fails
- **Error Handling**: Graceful degradation with informative error messages
- **Retry Mechanism**: Built-in retry logic for transient failures

### 4. **Easy Configuration**
- **Centralized Config**: All settings in `hybrid_config.py`
- **Runtime Adjustable**: Can modify weights and parameters without code changes
- **Performance Tuning**: Configurable timeouts and batch sizes

## 🚀 Usage

### Basic Usage
```python
from rag.rag_pipeline import rag_answer

# Use hybrid retrieval (default)
answer, sources = rag_answer("How to connect Snowflake to Atlan?")

# Use simple retrieval only
answer, sources = rag_answer("How to connect Snowflake to Atlan?", use_hybrid=False)
```

### Advanced Usage
```python
from rag.hybrid_retriever import create_hybrid_retriever
from langchain.chains import create_retrieval_chain

# Create custom hybrid retriever
retriever = create_hybrid_retriever()

# Use with custom chain
chain = create_retrieval_chain(retriever, document_chain)
result = chain.invoke({"input": "Your query here"})
```

## ⚙️ Configuration

Edit `rag/hybrid_config.py` to adjust settings:

```python
HYBRID_CONFIG = {
    "dense_weight": 0.7,      # Weight for semantic search
    "sparse_weight": 0.3,     # Weight for keyword search
    "top_k_dense": 5,         # Documents from dense search
    "top_k_sparse": 5,        # Documents from sparse search
    "final_top_k": 3,         # Final documents after merging
    "dedup_threshold": 0.8    # Deduplication threshold
}
```

## 🧪 Testing

Run the test script to validate the implementation:

```bash
python rag/test_hybrid.py
```

This will test both hybrid and simple retrieval methods with various query types.

## 📊 Benefits

### 1. **Improved Coverage**
- **Semantic Understanding**: Catches conceptual meaning and context
- **Exact Matching**: Ensures technical terms and error codes are found
- **Comprehensive Results**: Combines strengths of both approaches

### 2. **Better Accuracy**
- **Reduced False Negatives**: Less likely to miss relevant documents
- **Enhanced Precision**: Better ranking through combined scoring
- **Domain-Specific**: Handles both general and technical queries well

### 3. **Robustness**
- **Fault Tolerance**: Continues working even if one method fails
- **Graceful Degradation**: Falls back to working method
- **Error Recovery**: Built-in retry and error handling

## 🔍 How It Works

### 1. **Query Processing**
- Input query is processed by both dense and sparse retrieval methods
- Dense search uses ChromaDB similarity search with embeddings
- Sparse search uses BM25 scoring with tokenized text

### 2. **Result Combination**
- Scores are normalized to 0-1 range for fair comparison
- Weighted combination: `combined_score = dense_weight * dense_score + sparse_weight * sparse_score`
- Results are sorted by combined score

### 3. **Deduplication**
- Content-based hashing to identify duplicate documents
- Similarity threshold prevents near-duplicates
- Preserves highest-scoring version of each unique document

### 4. **Final Ranking**
- Top-k documents selected based on combined scores
- Results passed to LangChain for answer generation
- Source attribution maintained throughout the pipeline

## 🛠️ Troubleshooting

### Common Issues

1. **BM25 Index Build Failure**
   - Ensure documents are loaded correctly
   - Check that ChromaDB is populated
   - Verify `rank-bm25` package is installed

2. **Memory Issues**
   - Reduce `top_k_dense` and `top_k_sparse` values
   - Consider document chunking for large datasets
   - Monitor memory usage during indexing

3. **Performance Issues**
   - Enable parallel retrieval in config
   - Consider caching for frequent queries
   - Adjust timeout settings if needed

### Debug Mode

Enable detailed logging by setting:
```python
RETRIEVAL_PREFERENCES["log_retrieval_method"] = True
RETRIEVAL_PREFERENCES["show_scores"] = True
```

## 🔄 Migration from Simple RAG

The hybrid implementation is designed to be a drop-in replacement:

1. **No Breaking Changes**: Existing code continues to work
2. **Backward Compatibility**: Can still use simple retrieval
3. **Gradual Migration**: Enable hybrid retrieval when ready
4. **Easy Rollback**: Disable hybrid retrieval if issues arise

## 📈 Performance Considerations

- **Latency**: ~2-3x slower than simple retrieval due to dual search
- **Memory**: Additional memory for BM25 index
- **CPU**: Parallel processing helps minimize impact
- **Accuracy**: Significant improvement in retrieval quality

## 🎯 Best Practices

1. **Start with Default Config**: Use default settings initially
2. **Monitor Performance**: Track retrieval times and accuracy
3. **Tune Weights**: Adjust dense/sparse weights based on your data
4. **Test Thoroughly**: Validate with diverse query types
5. **Monitor Fallbacks**: Check logs for fallback frequency

## 🔮 Future Enhancements

- **Caching**: Result caching for improved performance
- **Async Processing**: True async retrieval for better concurrency
- **Dynamic Weights**: Adaptive weighting based on query type
- **More Retrieval Methods**: Integration with other retrieval techniques
- **A/B Testing**: Built-in comparison tools for different configurations

### Graph RAG Integration
- **Knowledge Graph Construction**: Extract entities and relationships from documentation
- **Graph-Enhanced Retrieval**: Use graph structure to improve context understanding
- **Multi-hop Reasoning**: Enable reasoning across related concepts and entities
- **Entity Linking**: Connect user queries to relevant entities in the knowledge graph
- **Relationship-Aware Responses**: Generate answers that leverage entity relationships
- **Graph-based Re-ranking**: Use graph centrality and connectivity for better ranking