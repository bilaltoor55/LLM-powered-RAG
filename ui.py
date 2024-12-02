import os
import time
import re
import streamlit as st
from langchain_community.llms import Ollama
from document_loader import load_documents_into_database
from models import get_list_of_models
from llm import getStreamingChain

# Define a persistent upload folder
UPLOAD_FOLDER = "/root/LLM-powered-RAG/Research"

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Constants
EMBEDDING_MODEL = "nomic-embed-text"

# Streamlit page configuration
st.set_page_config(
    page_title="Origen-LLM Demo",
    page_icon="Origen_blue.png",  # Replace with the name of your icon file
    layout="wide",
)

# Title
st.markdown("<h2 style='text-align: center;'><em>LLM RAG Assistant</em></h2>", unsafe_allow_html=True)

# Sidebar SVG Logo
svg_logo = """
<div style="margin-bottom: 20px;">
<!-- Include your SVG logo code here -->
</div>
"""
st.sidebar.markdown(svg_logo, unsafe_allow_html=True)

# Sidebar: Upload Documents Section
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose files", type=["pdf", "txt", "docx"], accept_multiple_files=True
)

# Toast message functionality
def show_toast(message, duration=3):
    """Display a toast-like message temporarily."""
    with st.empty():
        st.success(message)
        time.sleep(duration)

# Handle file uploads
uploaded_filenames = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        unique_filename = f"{int(time.time())}_{uploaded_file.name}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        # Save each file to the persistent upload folder
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_filenames.append(file_path)

    # Display a temporary toast message for successful upload
    show_toast(f"Uploaded {len(uploaded_files)} file(s) successfully!", duration=3)

# Sidebar: Model Selection
if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()

# Custom model names
model_names = {
    "mistral:latest": "Mistral by Meta",
    "gemma2:latest": "Gemma by Google",
    "llama3.2:latest": "Llama by OpenAI",
}

# Replace model names in the list with custom names
model_display_names = [model_names.get(model, model) for model in st.session_state["list_of_models"]]

selected_model = st.sidebar.selectbox(
    "Select a model:", model_display_names
)

# Get the actual model name from the selected display name
selected_model_name = [key for key, value in model_names.items() if value == selected_model][0]

# Set up the selected LLM model
if st.session_state.get("ollama_model") != selected_model_name:
    st.session_state["ollama_model"] = selected_model_name
    st.session_state["llm"] = Ollama(model=selected_model_name)

# Sidebar: Index Documents Button
if st.sidebar.button("Index Documents"):
    if uploaded_filenames:
        with st.spinner("Indexing the uploaded files..."):
            st.session_state["db"] = None  # Reset the database
            for file_path in uploaded_filenames:
                st.session_state["db"] = load_documents_into_database(
                    EMBEDDING_MODEL, file_path
                )
        # Display success message after indexing
        show_toast("All Set to Answer Questions", duration=3)
    else:
        st.sidebar.error("No files uploaded. Please upload files first!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to handle streamed responses
def display_response(prompt):
    """Stream and display the response dynamically."""
    # Add user query to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ensure database is loaded
    if "db" in st.session_state and st.session_state["db"] is not None:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = ""  # Initialize response buffer
                response_container = st.empty()  # Placeholder for dynamic updates

                # Generate and stream the response
                try:
                    stream = getStreamingChain(
                        prompt,
                        st.session_state["messages"],
                        st.session_state["llm"],
                        st.session_state["db"],
                    )
                    for chunk in stream:
                        # Clean each chunk by trimming spaces and replacing multiple spaces with a single one
                        clean_chunk = re.sub(r'\s+', ' ', chunk.strip())  # Replace multiple spaces with a single space
                        response_text += clean_chunk + " "  # Append cleaned chunk
                        response_container.markdown(response_text.strip())  # Update display

                    # Add final response to chat history
                    final_response = response_text.strip()  # Clean final response
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": final_response}
                    )
                    response_container.markdown(final_response)  # Ensure final render
                except Exception as e:
                    st.error(f"An error occurred while processing your request: {str(e)}")
    else:
        st.sidebar.error("Please upload and index documents before asking questions.")

# Chat Input Section
if prompt := st.chat_input("Ask me anything!"):
    display_response(prompt)
