import os
os.environ.pop("HF_HOME", None)  # Ensure default cache is used

from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

# Test loading the embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    print("Embeddings loaded successfully!")
    # Quick test
    test_embedding = embeddings.embed_query("This is a test sentence.")
    print(f"Embedding shape: {len(test_embedding)}")
except Exception as e:
    print(f"Error: {e}")