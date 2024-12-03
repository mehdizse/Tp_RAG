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
from fpdf import FPDF


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
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

        # Add pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Embedding model for document retrieval
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.prompt_template = """
        À partir des documents suivants, générez un CV professionnel :
        
        Contexte : {context}
        
        Demande : {query}
        
        Fournissez une réponse concise et bien formatée mettant en avant les compétences et expériences clés.
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
            no_repeat_ngram_size=2,
            do_sample=True,  # Enable sampling
            top_k=50,        # Top-k sampling
            top_p=0.95,      # Top-p (nucleus) sampling
            repetition_penalty=1.2  # Penalize repetition
        )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response

def save_cv_as_pdf(cv_text, folder='generated_cv', filename='generated_cv.pdf'):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Split the text into lines and add them to the PDF
    for line in cv_text.split('\n'):
        pdf.multi_cell(0, 10, line.encode('latin-1', 'replace').decode('latin-1'))
    
    # Save the PDF to the specified folder
    pdf.output(os.path.join(folder, filename))

def main():
    # Ensure load_documents is called correctly
    documents = load_documents('data')
    vectorstore = create_vector_store(documents)
    
    cv_generator = CVGenerator(vectorstore)
    
    query = """"Rédige un CV professionnel en français pour un ingénieur logiciel ayant 2 ans d'expérience en développement web et mobile. Ce CV doit inclure les sections suivantes :

    Informations personnelles : Nom fictif, adresse, email, et numéro de téléphone.
    Résumé professionnel : Un paragraphe décrivant brièvement les compétences principales et les objectifs professionnels.
    Compétences techniques : Langages de programmation (JavaScript, Python, Dart, etc.), frameworks (React, Flutter), outils (Git, Docker), et compétences annexes (tests unitaires, CI/CD, UX/UI).
    Expériences professionnelles : Deux postes détaillant les missions principales, les technologies utilisées, et les résultats obtenus.
    Formation : Diplôme d'ingénieur en informatique ou équivalent.
    Projets personnels : Exemples de projets web et mobiles, avec des liens éventuels.
    Langues : Français (natif) et Anglais (intermédiaire ou avancé). Rends le format clair, organisé et prêt à l'emploi pour une candidature.""""
    generated_cv = cv_generator.generate_cv(query)
    
    print(generated_cv)
    
    # Save the generated CV as a PDF
    save_cv_as_pdf(generated_cv)

if __name__ == "__main__":
    main()