from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


# Function to load documents and initialize Chroma database
def load_documents_into_database(model_name: str, documents_path: str):
    # Load documents
    try:
        print(f"Loading documents from {documents_path}...")
        loader = PyPDFLoader(documents_path)
        raw_documents = loader.load()
        print(f"Number of raw documents: {len(raw_documents)}")
    except Exception as e:
        raise ValueError(f"Error loading documents: {e}")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Number of split documents: {len(documents)}")
    
    # Initialize embeddings
    try:
        embeddings = OpenAIEmbeddings(model=model_name)
    except Exception as e:
        raise ValueError(f"Error initializing embeddings: {e}")
    
    # Initialize Chroma database
    try:
        db = Chroma.from_documents(documents, embeddings)
        print("Chroma database initialized successfully.")
    except Exception as e:
        raise ValueError(f"Error initializing Chroma database: {e}")
    
    return db
