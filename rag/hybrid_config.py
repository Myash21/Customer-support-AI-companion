"""
Configuration for Hybrid RAG Pipeline
Easy to modify parameters for different use cases
"""

# Hybrid Retrieval Configuration
HYBRID_CONFIG = {
    # Weighting between dense and sparse retrieval
    "dense_weight": 0.7,      # Higher weight for semantic understanding
    "sparse_weight": 0.3,     # Lower weight for exact keyword matching
    
    # Number of documents to retrieve from each method
    "top_k_dense": 5,         # Documents from dense search
    "top_k_sparse": 5,        # Documents from sparse search
    "final_top_k": 3,         # Final documents after merging
    
    # Deduplication settings
    "dedup_threshold": 0.8,   # Similarity threshold for deduplication
    
    # Fallback settings
    "enable_fallback": True,  # Enable fallback to simple retrieval if hybrid fails
    "max_retries": 3,         # Maximum retry attempts
}

# Retrieval Method Preferences
RETRIEVAL_PREFERENCES = {
    "prefer_hybrid": True,    # Prefer hybrid over simple retrieval
    "log_retrieval_method": True,  # Log which method is being used
    "show_scores": False,     # Show retrieval scores in output (for debugging)
}

# Performance Settings
PERFORMANCE_CONFIG = {
    "enable_caching": False,  # Enable result caching (future enhancement)
    "parallel_retrieval": True,  # Run dense and sparse retrieval in parallel
    "timeout_seconds": 30,    # Timeout for retrieval operations
}

def get_config():
    """Get the complete configuration dictionary"""
    return {
        "hybrid": HYBRID_CONFIG,
        "preferences": RETRIEVAL_PREFERENCES,
        "performance": PERFORMANCE_CONFIG
    }

def update_config(section: str, key: str, value):
    """
    Update a specific configuration value
    
    Args:
        section: Configuration section ('hybrid', 'preferences', 'performance')
        key: Configuration key
        value: New value
    """
    if section == "hybrid":
        HYBRID_CONFIG[key] = value
    elif section == "preferences":
        RETRIEVAL_PREFERENCES[key] = value
    elif section == "performance":
        PERFORMANCE_CONFIG[key] = value
    else:
        raise ValueError(f"Invalid section: {section}. Must be 'hybrid', 'preferences', or 'performance'")

def print_config():
    """Print current configuration"""
    print("🔧 Hybrid RAG Configuration:")
    print("=" * 40)
    
    for section_name, section_config in get_config().items():
        print(f"\n📋 {section_name.upper()}:")
        for key, value in section_config.items():
            print(f"  {key}: {value}")

