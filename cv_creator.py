# cv_creator.py
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Ensure load_documents is correctly imported
from load_documents import load_documents
from vector_creation import create_vector_store

load_dotenv()  # Load environment variables

class CVGenerator:
    def __init__(self, vectorstore, temperature=0.3):
        """
        Initialize CV Generator with vector store and GPT-2
        """
        self.vectorstore = vectorstore
        self.temperature = temperature
        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Add pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Embedding model for document retrieval
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
       
        self.prompt_template = """
        Based on the following context documents, help generate a professional CV:
       
        Context: {context}
       
        Task: {query}
       
        Please provide a concise, well-formatted response that highlights key professional details.
        """
       
    def generate_cv(self, query, additional_context=None):
        """
        Generate CV using retrieved documents and GPT-2
        """
        # Retrieve relevant documents
        retrieved_docs = self.vectorstore.similarity_search(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
       
        # Add any additional context
        if additional_context:
            context += f"\n\nAdditional Context: {additional_context}"
       
        # Prepare the prompt
        prompt = self.prompt_template.format(context=context, query=query)
       
        # Encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
       
        # Generate the response
        outputs = self.model.generate(
            inputs, 
            max_length=1024, 
            num_return_sequences=1,
            temperature=self.temperature,
            no_repeat_ngram_size=2
        )
       
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
       
        return response

def main():
    # Ensure load_documents is called correctly
    documents = load_documents('data')
    vectorstore = create_vector_store(documents)
    
    cv_generator = CVGenerator(vectorstore)
    
    query = "Generate a CV for a software engineer with 5 years of experience."
    generated_cv = cv_generator.generate_cv(query)
    
    print(generated_cv)
    
    # Optionally, save the CV to a file
    with open('generated_cv.txt', 'w', encoding='utf-8') as f:
        f.write(generated_cv)

if __name__ == "__main__":
    main()