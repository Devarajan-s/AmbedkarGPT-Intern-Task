import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_qa():
    try:
        loader = TextLoader('speech.txt')
        documents = loader.load()
        if not documents:
            print("Error: Could not load speech.txt. Make sure the file exists and is not empty.")
            return
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        print("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Creating vector store...")
        vector_store = Chroma.from_documents(chunks, embeddings)
        print("Initializing Ollama LLM...")
        llm = Ollama(model="mistral")
        retriever = vector_store.as_retriever()
        prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. You must answer questions based *only* on the following context.
If the answer is not found in the context, simply say 'I don't know, the answer is not in the provided text.'
Do not use any external knowledge.

Context: {context}

Question: {question}

Answer:""")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        
        print("\nAmbedkarGPT is ready.")
        print("Ask questions based on the speech. Type 'exit' to quit.")
        
        while True:
            user_query = input("\nQuestion: ")
            if user_query.lower().strip() == 'exit':
                print("Exiting...")
                break
            
            if not user_query.strip():
                continue
            response = rag_chain.invoke(user_query)

            print(f"Answer: {response}")

    except ImportError as e:
        print(f"\nImport Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    run_rag_qa()