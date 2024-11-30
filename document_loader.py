import os
from langchain.vectorstores import Chroma
from langchain.embeddings import NomicEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document

# Path to store the Chroma database
DATABASE_FOLDER = "/root/LLM-powered-RAG/chroma_db"

# Ensure the database folder exists
if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)


def load_document(file_path):
    """
    Load a single document based on its file type.
    Supports PDF, TXT, and DOCX formats.
    """
    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    return loader.load()


def split_document(document):
    """
    Split a document into smaller chunks for embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_documents(document)


def load_documents_into_database(embedding_model, document_path):
    """
    Load documents into the Chroma database.
    :param embedding_model: The embedding model to use (e.g., "nomic-embed-text").
    :param document_path: The path to the document to be loaded.
    """
    embeddings = NomicEmbeddings(embedding_model)
    documents = load_document(document_path)
    split_docs = split_document(documents)

    # Load documents into the Chroma database and persist
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=DATABASE_FOLDER,
    )
    db.persist()
    print(f"Documents successfully loaded into the database at {DATABASE_FOLDER}")


def get_database():
    """
    Initialize and return the Chroma database.
    :return: Chroma database object.
    """
    embeddings = NomicEmbeddings("nomic-embed-text")  # Ensure the same embedding model is used
    try:
        return Chroma(persist_directory=DATABASE_FOLDER, embedding_function=embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize database: {e}")


if __name__ == "__main__":
    # Example for testing
    test_file_path = "/path/to/test/document.pdf"  # Replace with your test file path
    embedding_model = "nomic-embed-text"

    # Load and test the document database
    try:
        load_documents_into_database(embedding_model, test_file_path)
        db = get_database()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error: {e}")
