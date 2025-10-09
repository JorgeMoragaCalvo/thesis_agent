import streamlit as st
import requests
from typing import List, Dict

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Knowledge Base - Testing Interface",
    page_icon=":robot_face:",
    layout="wide"
)

def upload_document(file) -> Dict:
    """Upload a document to the API."""
    files = {"file": (file.name, file, file.type)}
    response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
    return response.json()

def get_documents() -> List[Dict]:
    """Get the list of all documents."""
    response = requests.get(f"{API_BASE_URL}/documents")
    return response.json()

def delete_document(document_id: int):
    """Delete a document."""
    response = requests.delete(f"{API_BASE_URL}/documents/{document_id}")
    return response.status_code == 204

def query_knowledge_base(query: str, top_k: int = 5, threshold: float = 0.7) -> Dict:
    """Query the knowledge base."""
    payload = {
        "query": query,
        "top_k": top_k,
        "similarity_threshold": threshold
    }
    response = requests.post(f"{API_BASE_URL}/rag/query", json=payload)
    return response.json()

def check_health() -> Dict:
    """Check the health of the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.json()
    except:
        return {"status": "unavailable"}

# Main UI
st.title("📚 RAG Knowledge Base - Testing Interface")

with st.sidebar:
    st.header("System Status")
    health = check_health()

    if health.get("status") == "healthy":
        st.success("✅ API is running")
        st.success("✅ Database connected" if health.get("database_connected") else "❌ Database disconnected")
    else:
        st.error("❌ API unavailable")

    st.info(f"Version: {health.get('version', 'N/A')}")

    st.divider()

    st.header("Query Settings")
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5)
    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🔍 Query", "📄 Documents", "📤 Upload"])

# Tab 1: Query Interface
with tab1:
    st.header("Query the Knowledge Base")

    query_input = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What is linear programming?"
    )

    if st.button("Search", type="primary", use_container_width=True):
        if query_input:
            with st.spinner("Searching knowledge base..."):
                try:
                    result = query_knowledge_base(query_input, top_k, threshold)

                    # Display answer
                    st.subheader("Answer")
                    st.write(result["answer"])

                    # Display metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Response Time", f"{result['response_time']:.2f}s")
                    with col2:
                        st.metric("Chunks Retrieved", len(result["retrieved_chunks"]))

                    if result["retrieved_chunks"]:
                        st.divider()
                        st.subheader("Retrieved Context")

                        for idx, chunk in enumerate(result["retrieved_chunks"], 1):
                            with st.expander(
                                    f"📄 Chunk {idx} - {chunk['filename']} "
                                    f"(Similarity: {chunk['similarity_score']:.3f})"
                            ):
                                st.text(chunk['chunk_text'])
                                st.caption(f"Document ID: {chunk['document_id']} | "
                                    f"Chunk Index: {chunk['chunk_index']}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question.")

# Tab 2: Document Management
with tab2:
    st.header("Document Management")

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    try:
        documents = get_documents()

        if documents:
            st.write(f"Total documents: {len(documents)}")

            for doc in documents:
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    st.write(f"**{doc['filename']}**")
                with col2:
                    st.write(f"Type: {doc['file_type']}")
                with col3:
                    st.write(f"Chunks: {doc.get('chunk_count', 0)}")
                with col4:
                    if st.button("🗑️", key=f"delete_{doc['id']}"):
                        if delete_document(doc['id']):
                            st.success("Document deleted successfully.")
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")
                st.caption(f"Uploaded: {doc['created_at']}")
                st.divider()
        else:
            st.info("No documents uploaded yet. Upload some documents to get started!")
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

# Tab 3: Upload Interface
with tab3:
    st.header("Upload a Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT"
    )

    if uploaded_file:
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")

        if st.button("Upload", type="primary", use_container_width=True):
            with st.spinner("Uploading and processing document..."):
                try:
                    result = upload_document(uploaded_file)
                    st.success("✅ Document uploaded successfully!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Document ID", result["document_id"])
                    with col2:
                        st.metric("Chunks Created", result["chunks_created"])

                    st.info(result["message"])
                except Exception as e:
                    st.error(f"Error uploading document: {str(e)}")

# Footer
st.divider()
st.caption("RAG Knowledge Base Testing Interface | Optimization AI Tutor")