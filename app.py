import streamlit as st
from rag_pipeline import build_qa_chain

st.set_page_config(page_title="AI Document Chatbot")
st.title("📄 AI Chatbot (RAG)")
st.write("Ask questions from your documents")

@st.cache_resource
def load_chain():
    return build_qa_chain()

qa_chain = load_chain()

query = st.text_input("Enter your question:")

if query:
    try:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
            st.success("Answer:")
            st.write(answer)
    except Exception as e:
        st.error(f"Error: {str(e)}")
