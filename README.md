# CV Generator using RAG-based GPT-2 Model

## Overview

This project utilizes a Retrieval-Augmented Generation (RAG) based approach with GPT-2 to generate a CV from your own documents. The model retrieves relevant information from your documents and generates a coherent CV based on the provided prompts.

## Features

- **Document Loading**: Load and process PDF or text documents containing CV information.
- **RAG-based Generation**: Use a RAG-based GPT-2 model to generate a CV from the loaded documents.
- **Customizable Prompts**: Provide different prompts to generate various types of CVs.
- **Temperature Control**: Adjust the temperature parameter to control the randomness of the generated text.

## Installation

### Prerequisites

- Python 3.8 or later
- PyTorch
- Transformers
- Sentence-BERT
- LangChain
- Chroma
- dotenv (for environment variables)

### Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/cv-generator.git
    cd cv-generator
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**:
    Create a `.env` file in the project root directory and add any necessary environment variables, such as paths to your document directories.

## Usage

### Loading Documents

Place your old CV documents (in PDF or text format) in a directory. The script will load and process these documents to extract relevant information.

### Generating a CV

1. **Initialize the Generator**:
    ```python
    from cv_generator import CVGenerator
    from load_documents import load_documents
    from vector_creation import create_vector_store

    # Load documents
    documents = load_documents('path/to/your/documents')

    # Create vector store
    vectorstore = create_vector_store(documents)

    # Initialize CV generator
    cv_generator = CVGenerator(vectorstore)
    ```

2. **Generate a CV**:
    ```python
    query = "Generate a CV for a software engineer with 5 years of experience."
    generated_cv = cv_generator.generate_cv(query)
    print(generated_cv)

    # Optionally, save the CV to a file
    with open('generated_cv.txt', 'w', encoding='utf-8') as f:
        f.write(generated_cv)
    ```

### Customizing Prompts and Temperature

You can customize the prompts and temperature to generate different types of CVs.

- **Prompts**: Modify the `query` variable to change the input for the CV generation.
- **Temperature**: Adjust the `temperature` parameter in the `CVGenerator` class to control the randomness of the generated text.

## Example Prompt

```python
query = "Générer un CV pour un ingénieur logiciel avec:
- Compétences: Python, Machine Learning, SQL
- Expérience: 5 ans"
```

## Generation Parameters

The text generation parameters are set to ensure coherent and relevant text generation. The parameters used in the code are:

```python
outputs = self.model.generate(
    inputs, 
    max_length=1024, 
    num_return_sequences=1,
    temperature=self.temperature,
    no_repeat_ngram_size=2
)
```