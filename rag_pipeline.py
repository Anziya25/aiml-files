import os
os.environ["HF_HOME"] = "C:\\new_hf_cache"

import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer as Tokenizer,
    AutoModel
)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA


# ============================================================
# Custom Embeddings Class
# ============================================================
class CustomHuggingFaceEmbeddings:
    def __init__(self, model_name="albert-base-v2"):
        self.tokenizer = Tokenizer.from_pretrained(
            model_name,
            use_safetensors=False
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=False
        )

    def embed_query(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = (
                outputs.last_hidden_state
                .mean(dim=1)
                .squeeze()
                .numpy()
            )

        return embedding.tolist()

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


# ============================================================
# Load Documents
# ============================================================
def load_documents(folder_path=r"C:\ai_chatbot_rag\document"):

    docs = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    for file in os.listdir(folder_path):

        path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents")

    return docs


# ============================================================
# Split Documents
# ============================================================
def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    return chunks


# ============================================================
# Create Vector Store (Chroma)
# ============================================================
def create_vectorstore(chunks):

    print("Loading embeddings and creating Chroma vectorstore")

    embeddings = CustomHuggingFaceEmbeddings()

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# ============================================================
# Load LLM (T5-Small)
# ============================================================
def load_llm():

    print("Loading T5-Small")

    model_name = "t5-small"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_safetensors=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        use_safetensors=False
    )

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.3,
        device=0 if torch.cuda.is_available() else -1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


# ============================================================
# Build QA Chain
# ============================================================
def build_qa_chain():

    print("\nBuilding RAG Pipeline\n")

    docs = load_documents()
    chunks = split_documents(docs)

    vectorstore = create_vectorstore(chunks)
    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        return_source_documents=True
    )

    print("\nRAG Ready\n")

    return qa_chain
