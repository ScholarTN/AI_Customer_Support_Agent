import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import tempfile
import time


load_dotenv()


try:
    from streamlit_chat import message
    STREAMLIT_CHAT_AVAILABLE = True
except ImportError:
    STREAMLIT_CHAT_AVAILABLE = False
    st.warning("streamlit-chat not available. Using basic chat interface.")

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    st.warning("duckduckgo_search not available. Web search disabled.")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    st.warning("Pillow not available. Image processing disabled.")


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    st.error("Hugging Face API token not found. Please configure it in the environment.")
    st.stop()

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "company_name" not in st.session_state:
    st.session_state.company_name = ""
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []
if "company_registered" not in st.session_state:
    st.session_state.company_registered = False
if "processing_docs" not in st.session_state:
    st.session_state.processing_docs = False

# Configure page
st.set_page_config(
    page_title="AI Customer Support Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
}
.company-card {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.stButton button:hover {
    background-color: #45a049;
}
.chat-container {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 1rem;
    height: 600px;
    overflow-y: auto;
}
.info-box {
    background-color: #e8f4fd;
    border-left: 4px solid #1f77b4;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
}
.user-message {
    background-color: #e1f5fe;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    max-width: 80%;
    margin-left: auto;
}
.bot-message {
    background-color: #f3e5f5;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    max-width: 80%;
}
.progress-container {
    margin: 1rem 0;
    padding: 1rem;
    background-color: #f0f2f6;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Function to load documents
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path)
            elif uploaded_file.name.endswith('.docx'):
                loader = Docx2txtLoader(tmp_file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                continue
            
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass  # Ignore errors if file doesn't exist
    
    return documents

# Function to create vector store with progress tracking
def create_vectorstore(documents, company_name, progress_bar, status_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    status_text.text("Splitting documents into chunks...")
    texts = text_splitter.split_documents(documents)
    progress_bar.progress(0.3)
    
    status_text.text("Creating embeddings...")
   
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=HUGGINGFACEHUB_API_TOKEN)
    progress_bar.progress(0.6)
    
    status_text.text("Building vector store...")
    
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=f"vectorstore/{company_name}")
    progress_bar.progress(0.8)
    
    # Save vectorstore
    vectorstore.persist()
    progress_bar.progress(1.0)
    
    status_text.text("Complete!")
    time.sleep(0.5)  # Show completion for a moment
    
    return vectorstore

# Function to search the web for company information
def search_web(company_name, query):
    if not DDGS_AVAILABLE:
        return []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{company_name} {query}", max_results=3))
        return results
    except Exception as e:
        st.error(f"Error searching web: {e}")
        return []

# Function to initialize QA chain with a free model
def initialize_qa_chain(vectorstore):
    
    try:
        
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.1, "max_length": 512},
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
        )
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return None
    
    # Create custom prompt
    prompt_template = """You are a helpful AI assistant for customer support. 
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer based on the context, just say that you don't know. 
    Don't try to make up an answer. If the question is not related to the context, 
    politely respond that you are tuned to only answer questions that are related to the context.

    Context: {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain_type_kwargs = {"prompt": PROMPT}
    
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

# Function to get response from QA chain
def get_response(query, company_name):
    # First try to get answer from uploaded documents
    if st.session_state.qa_chain:
        try:
            result = st.session_state.qa_chain({"query": query})
            answer = result["result"]
            
            # If the answer is uncertain, search the web
            if "don't know" in answer.lower() or "not related" in answer.lower():
                web_results = search_web(company_name, query)
                if web_results:
                    answer += "\n\nI found some information from the web that might help:\n"
                    for i, res in enumerate(web_results, 1):
                        answer += f"{i}. {res['body']}\nSource: {res['href']}\n\n"
            return answer
        except Exception as e:
            st.error(f"Error getting response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    else:
        # If no documents are uploaded, search the web directly
        web_results = search_web(company_name, query)
        if web_results:
            answer = "I found some information from the web that might help:\n"
            for i, res in enumerate(web_results, 1):
                answer += f"{i}. {res['body']}\nSource: {res['href']}\n\n"
            return answer
        else:
            return "I couldn't find any information about this company. Please make sure the company name is correct or try uploading documents about the company."

# Function to display messages (fallback if streamlit-chat is not available)
def display_message(text, is_user=False):
    if STREAMLIT_CHAT_AVAILABLE:
        message(text, is_user=is_user, 
                avatar_style="adventurer" if is_user else "bottts", 
                seed=123)
    else:
        # Fallback basic message display
        if is_user:
            st.markdown(f'<div class="user-message"><strong>You:</strong> {text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><strong>Bot:</strong> {text}</div>', unsafe_allow_html=True)

# UI Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🤖 AI Customer Support Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("### Your intelligent assistant for company information and support")

# Information box
st.markdown("""
<div class="info-box">
    <strong>How to use:</strong> 
    <ol>
        <li>Enter your company name in the sidebar</li>
        <li>Upload documents about your company (PDF, TXT, DOCX)</li>
        <li>Click "Process Documents" to build your knowledge base</li>
        <li>Start asking questions about your company in the chat</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Sidebar for company registration and document upload
with st.sidebar:
    st.markdown('<h2 class="sub-header">Company Registration</h2>', unsafe_allow_html=True)
    
    # Company input with logo upload
    company_name = st.text_input("Company Name", value=st.session_state.company_name, placeholder="Enter company name")
    
    # Logo upload
    logo_file = st.file_uploader("Company Logo (optional)", type=["png", "jpg", "jpeg"])
    
    if company_name and company_name != st.session_state.company_name:
        st.session_state.company_name = company_name
        # Check if vectorstore exists for this company
        if os.path.exists(f"vectorstore/{company_name}"):
            try:
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(openai_api_key=HUGGINGFACEHUB_API_TOKEN)
                vectorstore = Chroma(persist_directory=f"vectorstore/{company_name}", embedding_function=embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = initialize_qa_chain(st.session_state.vectorstore)
                st.session_state.company_registered = True
                st.success(f"Loaded existing knowledge base for {company_name}")
            except Exception as e:
                st.error(f"Error loading vectorstore: {e}")
    
    st.markdown('<h2 class="sub-header">Upload Documents</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose documents (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files and company_name and not st.session_state.processing_docs:
        if st.button("Process Documents", key="process_btn"):
            st.session_state.processing_docs = True
            with st.spinner("Processing documents..."):
                # Create progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                documents = load_documents(uploaded_files)
                if documents:
                    st.session_state.vectorstore = create_vectorstore(
                        documents, company_name, progress_bar, status_text
                    )
                    st.session_state.qa_chain = initialize_qa_chain(st.session_state.vectorstore)
                    st.session_state.company_registered = True
                    st.success(f"Processed {len(documents)} documents for {company_name}!")
                else:
                    st.error("No valid documents could be processed.")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                st.session_state.processing_docs = False
    
    # Display company info if registered
    if st.session_state.company_registered:
        st.markdown("---")
        st.markdown(f"### Currently serving: {st.session_state.company_name}")
        if logo_file and PILLOW_AVAILABLE:
            st.image(logo_file, use_column_width=True)

# Main content area
if not st.session_state.company_registered:
    # Welcome screen if no company is registered
    st.info("👈 Please register a company in the sidebar to get started")
    
    # Display features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📄 Document Processing")
        st.write("Upload company documents to build a knowledge base")
    with col2:
        st.markdown("### 🔍 Smart Search")
        st.write("Get answers from both uploaded documents and web sources")
    with col3:
        st.markdown("### 💬 AI Chat")
        st.write("Natural conversations with AI-powered support")
else:
    # Chat interface
    st.markdown(f'<h2 class="sub-header">Chat with {st.session_state.company_name}</h2>', unsafe_allow_html=True)
    
    # Container for chat history
    chat_container = st.container()
    
    # Container for user input
    with st.form(key='input_form', clear_on_submit=True):
        user_input = st.text_input("Your question:", key='input', placeholder=f"Ask about {st.session_state.company_name}...")
        submit_button = st.form_submit_button(label='Send')
    
    # Generate response when input is submitted
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.past.append(user_input)
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = get_response(user_input, st.session_state.company_name)
            st.session_state.generated.append(response)
    
    # Display chat history
    if st.session_state.generated:
        with chat_container:
            for i in range(len(st.session_state.generated)):
                # User message
                display_message(st.session_state.past[i], is_user=True)
                
                # AI response
                display_message(st.session_state.generated[i], is_user=False)
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.generated = []
        st.session_state.past = []
        st.rerun()