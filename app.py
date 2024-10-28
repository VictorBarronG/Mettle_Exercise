from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
import streamlit as st
import fitz
import logging as logger
import httpx
import joblib

@st.cache_resource
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The file path of the PDF document to extract text from.

    Returns:
        str: A string containing all the extracted text from the PDF pages.
    """
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")  # Extract text from each page

        logger.info(f"Successfully extracted characters from {pdf_path}")
    return text

# Load the USCIS manual PDF file and extract text from it
pdf_path = "uscis_manual.pdf"
manual_text = extract_text_from_pdf(pdf_path)

# Split the extracted text into chunks - With more time this can be fine tuned
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
document_chunks = text_splitter.split_text(manual_text)

# Create the LLM and embeddings objects
llm = OllamaLLM(model="llama3.1", timeout=httpx.Timeout(10.0))
embeddings = OllamaEmbeddings(model='nomic-embed-text')

vector_store = FAISS.from_texts(document_chunks, embeddings)

def retrieve_relevant_chunks(query):
    """
    Retrieves relevant chunks from the USCIS manual based on a user query.

    This function takes a user query, embeds it using the Ollama embeddings,
    and then performs a similarity search using the FAISS vector store to find
    the top 6 most relevant chunks from the USCIS manual.

    Args:
        query (str): The user query to search for relevant chunks

    Returns:
        list: A list of relevant chunks from the USCIS manual
    """
    query_embedding = embeddings.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=5)
    return results

def main():
    """
    Streamlit application that allows users to ask questions about the USCIS manual
    and returns a concise and informative answer based on relevant passages from the manual.
    """
    st.title("USCIS Chatbot")

    user_query = st.text_input("Ask your question about USCIS")

    if user_query:
        relevant_chunks = retrieve_relevant_chunks(user_query)

        # Create a prompt for the LLM to generate an answer
        prompt = f"Here are some relevant passages from the USCIS manual: \n{relevant_chunks}\n\nBased on this information, please provide a concise and informative answer to the query: {user_query}"

        # Generate an answer using the LLM
        LLMresponse = llm.invoke(prompt)

        # Create a heading for the response
        response = f"Here's what I found about your query: \n\n{LLMresponse}"

        # Display the response to the user
        st.write(response)

if __name__ == "__main__":
    main()
