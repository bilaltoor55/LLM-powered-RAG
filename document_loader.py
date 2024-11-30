from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def load_documents_into_database(model_name: str, documents_path: str) -> Chroma:
    """
    Loads documents into the Chroma database and validates the process.

    Returns:
        Chroma: The Chroma database with loaded documents.

    Raises:
        ValueError: If no documents are loaded or database creation fails.
    """
    print("Loading documents")
    raw_documents = load_documents(documents_path)
    if not raw_documents:
        raise ValueError(f"No documents found in the provided path: {documents_path}")

    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Creating embeddings and loading documents into Chroma")
    db = Chroma.from_documents(
        documents,
        OllamaEmbeddings(model=model_name),
    )
    if db is None:
        raise ValueError("Failed to create Chroma database. Ensure valid documents and embeddings.")
    
    return db



def load_documents(path: str) -> List[Document]:
    """
    Loads documents from the specified path, which can be either a directory or a single file.

    Args:
        path (str): The path to the directory or file containing documents to load.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    docs = []

    # If the path is a directory, use DirectoryLoader
    if os.path.isdir(path):
        loaders = {
            ".pdf": DirectoryLoader(
                path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True,
            ),
            ".md": DirectoryLoader(
                path,
                glob="**/*.md",
                loader_cls=TextLoader,
                show_progress=True,
            ),
        }

        for file_type, loader in loaders.items():
            print(f"Loading {file_type} files from directory: {path}")
            docs.extend(loader.load())

    # If the path is a single file, handle file types directly
    elif os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            print(f"Loading single PDF file: {path}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif ext == ".md":
            print(f"Loading single Markdown file: {path}")
            loader = TextLoader(path)
            docs.extend(loader.load())
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        raise ValueError(f"Invalid path: {path}")

    return docs
