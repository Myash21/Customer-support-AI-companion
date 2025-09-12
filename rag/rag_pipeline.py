from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
try:
    from .hybrid_retrieval_adapter import create_hybrid_retriever
except ImportError:
    from hybrid_retrieval_adapter import create_hybrid_retriever
load_dotenv()

# -------------------------
# Load Vector DB
# -------------------------
def load_vector_db(persist_dir="chroma_db"):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name="db_docs",  # Use the correct collection name
        persist_directory=persist_dir, 
        embedding_function=embedding
    )
    return vectordb

# -------------------------
# Load Gemini LLM
# -------------------------
def load_gemini_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        max_output_tokens=2048,  # Increased token limit
        max_retries=3,  # Add retry mechanism
        request_timeout=60  # Add timeout
    )
    return llm

# -------------------------
# Note: Using vectordb.as_retriever() directly instead of custom retriever
# -------------------------

# -------------------------
# Hybrid RAG Pipeline (Dense + Sparse Retrieval)
# -------------------------
def build_hybrid_rag_pipeline():
    """Build a hybrid RAG pipeline using dense + sparse retrieval"""
    try:
        llm = load_gemini_model()
        
        # Create hybrid retriever (combines dense + sparse search)
        retriever = create_hybrid_retriever()
        
        # Create prompt template with more explicit instructions
        prompt_template = """You are Atlan's customer support assistant. Use only the Context to answer the Question.

Guidelines:
- Cite relevant facts from the Context; do not invent.
- If the Context is insufficient, state that clearly and list what is missing.
- Prefer concise, actionable steps, prerequisites, and exact values when present.
- Include role or permission names, configuration paths, and setting values if provided in Context.
- Expand acronyms on first use when helpful.
- If multiple paths exist, enumerate options with brief pros/cons.
- End with Next steps the user can take.

Context:
{context}

Question:
{input}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, PROMPT)
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain, retriever
        
    except Exception as e:
        print(f"Error building hybrid RAG pipeline: {e}")
        # Fallback to simple RAG pipeline if hybrid fails
        print("Falling back to simple RAG pipeline...")
        return build_simple_rag_pipeline()

# -------------------------
# Simple RAG Pipeline (Fallback)
# -------------------------
def build_simple_rag_pipeline():
    """Build a simplified RAG pipeline using create_retrieval_chain (fallback method)"""
    try:
        vectordb = load_vector_db()
        llm = load_gemini_model()
        
        # Create retriever from vectordb with more conservative settings
        retriever = vectordb.as_retriever(
            search_kwargs={"k": 3},  # Reduced from 5 to 3 to avoid overwhelming the LLM
            search_type="similarity"
        )
        
        # Create prompt template with more explicit instructions
        prompt_template = """You are Atlan's customer support assistant. Use only the Context to answer the Question.

Guidelines:
- Cite relevant facts from the Context; do not invent.
- If the Context is insufficient, state that clearly and list what is missing.
- Prefer concise, actionable steps, prerequisites, and exact values when present.
- Include role or permission names, configuration paths, and setting values if provided in Context.
- Expand acronyms on first use when helpful.
- If multiple paths exist, enumerate options with brief pros/cons.
- End with Next steps the user can take.

Context:
{context}

Question:
{input}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, PROMPT)
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain, vectordb
        
    except Exception as e:
        print(f"Error building RAG pipeline: {e}")
        raise e

# -------------------------
# Simple Query Function
# -------------------------
def rag_answer(query: str, max_retries: int = 3, use_hybrid: bool = True):
    """
    RAG answer function with hybrid retrieval and fallback mechanism
    
    Args:
        query: User query
        max_retries: Maximum number of retry attempts
        use_hybrid: Whether to use hybrid retrieval (dense + sparse) or fallback to simple
    """
    for attempt in range(max_retries):
        try:
            # Try hybrid retrieval first, fallback to simple if requested or if hybrid fails
            if use_hybrid:
                try:
                    retrieval_chain, retriever = build_hybrid_rag_pipeline()
                    print("✅ Using hybrid retrieval (dense + sparse)")
                except Exception as e:
                    print(f"⚠️ Hybrid retrieval failed: {e}")
                    print("Falling back to simple retrieval...")
                    retrieval_chain, retriever = build_simple_rag_pipeline()
                    print("✅ Using simple retrieval (dense only)")
            else:
                retrieval_chain, retriever = build_simple_rag_pipeline()
                print("✅ Using simple retrieval (dense only)")
            
            # Get answer using the chain structure
            result = retrieval_chain.invoke({"input": query})
            
            # Extract answer and context from the structure
            answer = result.get("answer", "")
            context = result.get("context", [])
            
            # Validate response
            if not answer or len(answer.strip()) < 10:
                print(f"Attempt {attempt + 1}: Empty or too short response, retrying...")
                if attempt < max_retries - 1:
                    continue
                else:
                    # Last attempt failed, return a fallback response
                    answer = "I apologize, but I'm having trouble generating a response. Please try rephrasing your question or contact support for assistance."
            
            # Get sources from context documents
            sources = []
            for doc in context:
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
            
            print(f"Success on attempt {attempt + 1}")
            return answer, sources
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 2}/{max_retries})")
                continue
            else:
                return f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)}", ["Error"]
    
    # This should never be reached, but just in case
    return "I apologize, but I'm unable to process your request at this time. Please try again later.", ["Error"]


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    query = "Connecting Snowflake to Atlan - required permissions? Hi team, we're trying to set up our primary Snowflake production database as a new source in Atlan, but the connection keeps failing. We've tried using our standard service account, but it's not working. Our entire BI team is blocked on this integration for a major upcoming project, so it's quite urgent. Could you please provide a definitive list of the exact permissions and credentials needed on the Snowflake side to get this working? Thanks."
    
    print("🔍 Testing Hybrid RAG Pipeline...")
    print("=" * 50)
    
    # Test hybrid retrieval
    print("\n1. Testing Hybrid Retrieval (Dense + Sparse):")
    answer, sources = rag_answer(query, use_hybrid=True)
    print("Answer:", answer)
    print("Sources:", sources)
    
    print("\n" + "=" * 50)
    
    # Test simple retrieval for comparison
    print("\n2. Testing Simple Retrieval (Dense only):")
    answer_simple, sources_simple = rag_answer(query, use_hybrid=False)
    print("Answer:", answer_simple)
    print("Sources:", sources_simple)
