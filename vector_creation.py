from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(documents, persist_directory='./chroma_db'):
    """
    Create a vector store from documents
    """
    # Use a pre-trained embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents, 
        embeddings, 
        persist_directory=persist_directory
    )
    return vectorstore

def retrieve_relevant_docs(vectorstore, query, k=3):
    """
    Retrieve most relevant documents for a query
    """
    return vectorstore.similarity_search(query, k=k)