import os
import time
import streamlit as st
from langchain_community.llms import Ollama
from document_loader import load_documents_into_database
from models import get_list_of_models
from llm import getStreamingChain

# Define a persistent upload folder
UPLOAD_FOLDER = "/home/root/uploads"  # Updated path for the droplet environment

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Constants
EMBEDDING_MODEL = "nomic-embed-text"

# Title
st.markdown("<h2 style='text-align: center;'>ORIGEN - LLM RAG Assistant</h2>", unsafe_allow_html=True)

# Sidebar SVG Logo
svg_logo = """<div style="margin-bottom: 20px;">
<svg xmlns="http://www.w3.org/2000/svg" width="250" height="100" viewBox="0 0 857 310" fill="none">
<g clip-path="url(#clip0_2051_176)"></g>
<defs>
<clipPath id="clip0_2051_176">
<rect width="330" height="100.547" fill="white" transform="translate(0 104.727)"/>
</clipPath>
</defs>
</svg></div>"""
st.sidebar.markdown(svg_logo, unsafe_allow_html=True)

# Upload Documents Section
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose files", type=["pdf", "txt", "docx"], accept_multiple_files=True
)

# Function to simulate a toast-like effect
def show_toast(message, duration=2):
    """Display a toast-like message temporarily."""
    with st.empty():
        st.success(message)
        time.sleep(duration)

# Handle file uploads
uploaded_filenames = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Add timestamp to filename for uniqueness
        unique_filename = f"{int(time.time())}_{uploaded_file.name}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        # Save each file to the persistent upload folder
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_filenames.append(file_path)

    # Display a temporary toast message for successful upload
    show_toast(f"Uploaded {len(uploaded_files)} file(s) successfully!")

# Model Selection
if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()

selected_model = st.sidebar.selectbox(
    "Select a model:", st.session_state["list_of_models"]
)

# Set up the selected LLM model
if st.session_state.get("ollama_model") != selected_model:
    st.session_state["ollama_model"] = selected_model
    st.session_state["llm"] = Ollama(model=selected_model)

# Index Documents Button
if st.sidebar.button("Index Documents"):
    if uploaded_filenames:
        with st.spinner("Indexing the uploaded files..."):
            for file_path in uploaded_filenames:
                load_documents_into_database(EMBEDDING_MODEL, file_path)
        st.sidebar.success("All uploaded documents have been indexed successfully!")
    else:
        st.sidebar.error("No files uploaded. Please upload files first!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and generate responses
if prompt := st.chat_input("Ask a question:"):
    # Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Generate response and stream
            stream = getStreamingChain(
                prompt,
                st.session_state.messages,
                st.session_state["llm"],
                st.session_state.get("db"),
            )
            response = st.empty()  # Placeholder for the response text
            for chunk in stream:
                response.markdown(chunk)

# Warning if no database is loaded
if "db" not in st.session_state:
    st.sidebar.warning("Please index documents to enable the chatbot.")
