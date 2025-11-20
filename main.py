"""
RAG-based Q&A System for Dr. B.R. Ambedkar's Speech
Uses LangChain, ChromaDB, HuggingFace Embeddings, and Ollama Mistral
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader

def load_document(file_path):
    """Load the speech text file."""
    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    print(f"Document loaded successfully! ({len(documents[0].page_content)} characters)")
    return documents

def split_text(documents, chunk_size=500, chunk_overlap=50):
    """Split the document into smaller chunks for better retrieval."""
    print(f"\nSplitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    return chunks

def create_vector_store(chunks, persist_directory="./chroma_db"):
    """Create embeddings and store them in ChromaDB."""
    print(f"\nCreating embeddings and vector store...")
    print("(This may take a minute on first run as the model downloads)")
    
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f" Vector store created with {len(chunks)} embedded chunks")
    return vectorstore

def load_existing_vector_store(persist_directory="./chroma_db"):
    """Load an existing ChromaDB vector store."""
    print(f"\nLoading existing vector store from {persist_directory}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    print("Vector store loaded successfully")
    return vectorstore

def setup_qa_chain(vectorstore):
    """Set up the QA chain with Ollama Mistral LLM."""
    print("\nSetting up QA chain with Ollama Mistral...")
    
    
    llm = Ollama(
        model="mistral",
        temperature=0.2  
    )
    
    
    prompt_template = """You are an AI assistant answering questions about Dr. B.R. Ambedkar's speech.
Use ONLY the context provided below to answer the question. If the answer cannot be found in the context, say "I cannot answer this based on the provided speech."

Context: {context}

Question: {question}

Answer: Let me provide a clear and accurate answer based on the speech:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print(" QA chain ready!")
    return qa_chain

def ask_question(qa_chain, question):
    """Ask a question and get an answer."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    
    result = qa_chain({"query": question})
    
    print(f"\nAnswer:\n{result['result']}")
    
    
    if result.get('source_documents'):
        print(f"\n[Retrieved {len(result['source_documents'])} relevant text chunks]")
    
    return result

def main():
    """Main function to run the RAG Q&A system."""
    print("="*60)
    print("RAG Q&A System - Dr. B.R. Ambedkar's Speech")
    print("="*60)
    
    speech_file = "speech.txt"
    persist_dir = "./chroma_db"
    
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("\n Found existing vector store!")
        use_existing = input("Use existing vector store? (y/n): ").strip().lower()
        
        if use_existing == 'y':
            vectorstore = load_existing_vector_store(persist_dir)
        else:
            print("\nCreating new vector store...")
            documents = load_document(speech_file)
            chunks = split_text(documents)
            vectorstore = create_vector_store(chunks, persist_dir)
    else:
        
        documents = load_document(speech_file)
        chunks = split_text(documents)
        vectorstore = create_vector_store(chunks, persist_dir)
    
    
    qa_chain = setup_qa_chain(vectorstore)
    
    print("\n" + "="*60)
    print("System ready! You can now ask questions.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*60)
    
    # Interactive Q&A loop
    while True:
        print("\n")
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Q&A system. ")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        try:
            ask_question(qa_chain, question)
        except Exception as e:
            print(f"\n Error: {str(e)}")
            print("Please make sure Ollama is running and the Mistral model is installed.")

if __name__ == "__main__":
    main()