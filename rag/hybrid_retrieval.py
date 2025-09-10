"""
Hybrid Retrieval Implementation
Combines dense (semantic) and sparse (BM25) retrieval methods
"""

import asyncio
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from collections import defaultdict
try:
    from .hybrid_config import HYBRID_CONFIG
except ImportError:
    from hybrid_config import HYBRID_CONFIG


class HybridRetrievalOrchestrator:
    """
    Orchestrates hybrid retrieval combining dense and sparse search methods
    Prioritizes dense retrieval with fallback to sparse retrieval
    """
    
    def __init__(self, vector_store: Chroma, documents: List[str], metadatas: List[Dict]):
        """
        Initialize hybrid retrieval orchestrator
        
        Args:
            vector_store: ChromaDB vector store for dense retrieval
            documents: List of document texts for BM25 indexing
            metadatas: List of metadata dictionaries corresponding to documents
        """
        self.vector_store = vector_store
        self.documents = documents
        self.metadatas = metadatas
        
        # Use configuration from hybrid_config.py
        self.config = HYBRID_CONFIG.copy()
        
        # Initialize BM25 index
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from documents"""
        try:
            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in self.documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            print("✅ BM25 index built successfully")
        except Exception as e:
            print(f"❌ Error building BM25 index: {e}")
            self.bm25_index = None
    
    def _dense_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Perform dense retrieval using ChromaDB
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents with scores from dense retrieval
        """
        try:
            # Use similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            dense_results = []
            for doc, score in results:
                dense_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "dense_score": float(score),
                    "sparse_score": 0.0,
                    "combined_score": 0.0,
                    "source": "dense"
                })
            
            return dense_results
        except Exception as e:
            print(f"❌ Dense search failed: {e}")
            return []
    
    def _sparse_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Perform sparse retrieval using BM25
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents with scores from sparse retrieval
        """
        try:
            if self.bm25_index is None:
                return []
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top-k documents
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            sparse_results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    sparse_results.append({
                        "content": self.documents[idx],
                        "metadata": self.metadatas[idx],
                        "dense_score": 0.0,
                        "sparse_score": float(scores[idx]),
                        "combined_score": 0.0,
                        "source": "sparse"
                    })
            
            return sparse_results
        except Exception as e:
            print(f"❌ Sparse search failed: {e}")
            return []
    
    def _normalize_scores(self, results: List[Dict], score_key: str) -> List[Dict]:
        """
        Normalize scores to 0-1 range
        
        Args:
            results: List of results with scores
            score_key: Key name for the score to normalize
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        scores = [r[score_key] for r in results if r[score_key] > 0]
        if not scores:
            return results
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            for result in results:
                result[f"normalized_{score_key}"] = 1.0
        else:
            for result in results:
                if result[score_key] > 0:
                    normalized = (result[score_key] - min_score) / (max_score - min_score)
                    result[f"normalized_{score_key}"] = normalized
                else:
                    result[f"normalized_{score_key}"] = 0.0
        
        return results
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate results based on content similarity
        
        Args:
            results: List of results to deduplicate
            
        Returns:
            Deduplicated results
        """
        if not results:
            return results
        
        # Group by content hash or similar content
        seen_content = set()
        deduplicated = []
        
        for result in results:
            content_hash = hash(result["content"][:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated.append(result)
        
        return deduplicated
    
    def _combine_and_rank(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """
        Combine and re-rank results from both retrieval methods
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            
        Returns:
            Combined and ranked results
        """
        # Normalize scores for both methods
        dense_results = self._normalize_scores(dense_results, "dense_score")
        sparse_results = self._normalize_scores(sparse_results, "sparse_score")
        
        # Create a combined results dictionary
        combined_results = {}
        
        # Add dense results
        for result in dense_results:
            doc_id = result["metadata"].get("content_hash", hash(result["content"]))
            combined_results[doc_id] = result
        
        # Add or update with sparse results
        for result in sparse_results:
            doc_id = result["metadata"].get("content_hash", hash(result["content"]))
            
            if doc_id in combined_results:
                # Document exists in both - combine scores
                existing = combined_results[doc_id]
                existing["sparse_score"] = result["sparse_score"]
                existing["normalized_sparse_score"] = result["normalized_sparse_score"]
                existing["source"] = "hybrid"
            else:
                # New document from sparse search
                combined_results[doc_id] = result
        
        # Calculate combined scores
        for result in combined_results.values():
            dense_score = result.get("normalized_dense_score", 0.0)
            sparse_score = result.get("normalized_sparse_score", 0.0)
            
            # Weighted combination (prioritizing dense retrieval)
            combined_score = (
                self.config["dense_weight"] * dense_score + 
                self.config["sparse_weight"] * sparse_score
            )
            result["combined_score"] = combined_score
        
        # Sort by combined score and return top results
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return final_results[:self.config["final_top_k"]]
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Main retrieval method that combines dense and sparse search
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            Combined and ranked results from both retrieval methods
        """
        if top_k is None:
            top_k = self.config["final_top_k"]
        
        # Perform both dense and sparse retrieval
        dense_results = self._dense_search(query, self.config["top_k_dense"])
        sparse_results = self._sparse_search(query, self.config["top_k_sparse"])
        
        # If dense search fails, fallback to sparse search
        if not dense_results and sparse_results:
            print("⚠️ Dense search failed, using sparse search results only")
            return sparse_results[:top_k]
        
        # If sparse search fails, use dense search only
        if not sparse_results and dense_results:
            print("⚠️ Sparse search failed, using dense search results only")
            return dense_results[:top_k]
        
        # If both fail, return empty results
        if not dense_results and not sparse_results:
            print("❌ Both dense and sparse search failed")
            return []
        
        # Combine and rank results
        combined_results = self._combine_and_rank(dense_results, sparse_results)
        
        # Deduplicate results
        final_results = self._deduplicate_results(combined_results)
        
        return final_results[:top_k]


def load_documents_for_bm25(persist_dir: str = "chroma_db") -> tuple:
    """
    Load documents and metadata from ChromaDB for BM25 indexing
    
    Args:
        persist_dir: ChromaDB persist directory
        
    Returns:
        Tuple of (documents, metadatas)
    """
    try:
        # Load the same embedding model used in the original pipeline
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load ChromaDB
        vectordb = Chroma(
            collection_name="db_docs",
            persist_directory=persist_dir,
            embedding_function=embedding
        )
        
        # Get all documents from the collection
        # Note: This is a workaround since ChromaDB doesn't have a direct way to get all documents
        # We'll use a broad query to get most documents
        all_docs = vectordb.similarity_search("", k=1000)  # Get up to 1000 documents
        
        documents = [doc.page_content for doc in all_docs]
        metadatas = [doc.metadata for doc in all_docs]
        
        print(f"✅ Loaded {len(documents)} documents for BM25 indexing")
        return documents, metadatas
        
    except Exception as e:
        print(f"❌ Error loading documents for BM25: {e}")
        return [], []
