"""
Custom LangChain Retriever for Hybrid Retrieval
Wraps the HybridRetrievalOrchestrator to work with LangChain
"""

from typing import List, Optional
from langchain.schema import BaseRetriever, Document
from langchain_chroma import Chroma
try:
    from .hybrid_retrieval_engine import HybridRetrievalOrchestrator, load_documents_for_bm25
except ImportError:
    from hybrid_retrieval_engine import HybridRetrievalOrchestrator, load_documents_for_bm25


class HybridRetriever(BaseRetriever):
    """
    Custom LangChain retriever that uses hybrid retrieval (dense + sparse)
    """
    
    def __init__(self, vector_store: Chroma, documents: List[str], metadatas: List[dict]):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: ChromaDB vector store for dense retrieval
            documents: List of document texts for BM25 indexing
            metadatas: List of metadata dictionaries corresponding to documents
        """
        super().__init__()
        self._hybrid_orchestrator = HybridRetrievalOrchestrator(
            vector_store=vector_store,
            documents=documents,
            metadatas=metadatas
        )
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents using hybrid retrieval
        
        Args:
            query: Search query
            
        Returns:
            List of relevant Document objects
        """
        # Get results from hybrid retrieval
        results = self._hybrid_orchestrator.retrieve(query)
        
        # Convert to LangChain Document format
        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata=result["metadata"]
            )
            documents.append(doc)
        
        return documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async version of document retrieval
        
        Args:
            query: Search query
            
        Returns:
            List of relevant Document objects
        """
        # For now, just call the sync version
        # In the future, this could be made truly async
        return self._get_relevant_documents(query)


def create_hybrid_retriever(persist_dir: str = "chroma_db") -> HybridRetriever:
    """
    Factory function to create a hybrid retriever
    
    Args:
        persist_dir: ChromaDB persist directory
        
    Returns:
        Configured HybridRetriever instance
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # Load vector store
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="db_docs",
        persist_directory=persist_dir,
        embedding_function=embedding
    )
    
    # Load documents for BM25
    documents, metadatas = load_documents_for_bm25(persist_dir)
    
    if not documents:
        raise ValueError("No documents found for BM25 indexing. Please ensure ChromaDB is populated.")
    
    # Create hybrid retriever
    retriever = HybridRetriever(vector_store, documents, metadatas)
    
    print("✅ Hybrid retriever created successfully")
    return retriever
