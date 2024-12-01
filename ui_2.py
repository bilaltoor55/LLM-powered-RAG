import os
import time
import re
import streamlit as st
from langchain_community.llms import Ollama
from document_loader import load_documents_into_database
from models import get_list_of_models
from llm import getStreamingChain

# Define a persistent upload folder
UPLOAD_FOLDER = "/home/root/uploads"

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Constants
EMBEDDING_MODEL = "nomic-embed-text"

st.set_page_config(
    page_title="Origen-LLM Demo",
    page_icon="Origen_blue.png",  # Replace with the name of your icon file
    layout="wide"
)

# Title
st.markdown("<h2 style='text-align: center;'>ORIGEN - LLM RAG Assistant</h2>", unsafe_allow_html=True)

# Sidebar SVG Logo
svg_logo = """<div style="margin-bottom: 20px;">
<svg xmlns="http://www.w3.org/2000/svg" width="250" height="100" viewBox="0 0 857 310" fill="none">
<g clip-path="url(#clip0_2051_176)">
<path d="M320.716 135.664V136.475H318.151V143.398H317.096V136.475H314.531V135.664H320.716Z" fill="#016DD3"/>
<path d="M321.451 143.398V135.664H322.789L325.782 142.117L328.719 135.664H35.136C754.912 205.892 753.94 206.27 752.68 206.27Z" fill="#016DD3"/>
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

# Model Mapping for Custom Display Names
model_mapping = {
    "Mistral by Google": "mistral:latest",
    "Gemma2 by Meta": "gemma2:latest",
    "Llama3.2 by OpenAI": "llama3.2:latest",
    "Nomic Embed Text": "nomic-embed-text:latest"
}

# Select Model from Sidebar
if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()

# Display the custom model names in the dropdown
selected_model_name = st.sidebar.selectbox(
    "Select a model:", list(model_mapping.keys())
)

# Map the selected model name to the actual model identifier
selected_model = model_mapping[selected_model_name]

# Set up the selected LLM model
if st.session_state.get("ollama_model") != selected_model:
    st.session_state["ollama_model"] = selected_model
    st.session_state["llm"] = Ollama(model=selected_model)

# Index Documents Button
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
                        # Clean each chunk to remove leading/trailing spaces and extra spaces between words
                        clean_chunk = re.sub(r'\s+', ' ', chunk.strip())  # Replace multiple spaces with a single one
                        response_text += clean_chunk + " "  # Append cleaned chunk
                        response_container.markdown(response_text.strip())  # Update display

                    # Add final response to chat history
                    final_response = response_text.strip()  # Clean final response
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": final_response}
                    )
                    response_container.markdown(final_response)  # Ensure final render
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        st.sidebar.warning("No database loaded. Please index documents first!")

# Handle user input and generate responses
prompt = st.chat_input("Ask a question:")  # Ensure the input bar is here
if prompt:
    display_response(prompt)

# Warning if no database is loaded
if "db" not in st.session_state:
    st.sidebar.warning("Please index documents to enable the chatbot.")
