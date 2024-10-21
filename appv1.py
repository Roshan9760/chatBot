import streamlit as st
import PyPDF2
import cohere
import chromadb

# Initialize Cohere and Chroma clients
COHERE_API_KEY = 'C1gKHRZE7Lp4vopHevghOMo5sdLCwDfOH31JFWNs'  #API key
co = cohere.Client(COHERE_API_KEY)
client = chromadb.Client()

# Helper function to extract text from PDF
def extract_and_split_text_from_pdf(pdf_file):
    """
    Extracts the text from the PDF file and splits it into sentences.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split the text into smaller sections (sentences for now)
    sections = text.split('. ')  # You can customize the splitting logic if needed
    return sections

# Helper function to generate embeddings using Cohere
def generate_embeddings(text_list):
    response = co.embed(texts=text_list)
    return response.embeddings

# Helper function to store embeddings in Chroma
def store_embeddings_in_chroma(text_list, embeddings):
    collection_name = "pdf_embeddings_collection"
    
    # Try to get the existing collection or create a new one
    try:
        collection = client.get_collection(collection_name)
        st.write(f"Using existing collection: {collection_name}")
    except:
        collection = client.create_collection(name=collection_name)
        st.write(f"Created new collection: {collection_name}")

    # Store embeddings
    for i, text in enumerate(text_list):
        collection.add(documents=[text], embeddings=[embeddings[i]], ids=[f'doc_{i}'])
    st.write("Embeddings stored in Chroma.")

# Helper function to search in Chroma and generate a well-structured answer using LLM
def search_in_chroma(query):
    # Generate embedding for the query
    query_embedding = generate_embeddings([query])[0]
    collection = client.get_collection("pdf_embeddings_collection")
    
    # Retrieve the most relevant document based on the query
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    if results and 'documents' in results:
        # Get the relevant document text
        relevant_text = results['documents'][0]

        # Use the LLM to generate a well-structured answer
        answer_response = co.generate(
            prompt=f"Based on the following information: '{relevant_text}', please provide a detailed answer to the question: '{query}'.",
            max_tokens=150,  # You can adjust this value
            temperature=0.7  # Adjust this for more creative or focused answers
        )
        return answer_response.generations[0].text.strip()
    else:
        return "No relevant information found."

# Streamlit App UI
st.title("PDF Chatbot using LLM and Chroma")

# Session state to hold uploaded PDF and processed content
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'text_sections' not in st.session_state:
    st.session_state.text_sections = None

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.write("Extracting and embedding the content from the PDF...")

    # Step 1: Extract text from PDF
    st.session_state.text_sections = extract_and_split_text_from_pdf(uploaded_file)

    # Step 2: Generate embeddings for the extracted text
    text_embeddings = generate_embeddings(st.session_state.text_sections)

    # Step 3: Store embeddings in Chroma
    store_embeddings_in_chroma(st.session_state.text_sections, text_embeddings)

    st.success("PDF content has been successfully processed and embedded.")

# Step 4: Allow the user to ask questions
if st.session_state.uploaded_file is not None:
    query = st.text_input("Ask a question based on the PDF content")

    if st.button("Submit Question"):
        if query.lower() == "exit":
            st.write("Exiting chatbot. Goodbye!")
            st.session_state.uploaded_file = None  # Reset the state if needed
        elif query:
            answer = search_in_chroma(query)
            st.write(f"Answer: {answer}")
        else:
            st.warning("Please enter a question.")
