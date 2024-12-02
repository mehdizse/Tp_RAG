from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(directory, filename))
        elif filename.endswith('.txt'):
            loader = TextLoader(os.path.join(directory, filename))
        else:
            continue
        documents.extend(loader.load())
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into smaller chunks for better embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)