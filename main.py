# Requirements: pip install streamlit langchain langchain-community faiss-cpu pypdf python-dotenv sentence-transformers huggingface-hub transformers torch

import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import pickle
from typing import List
import json
from datetime import datetime

# LangChain imports - Updated for newer versions
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Load environment variables and get HuggingFace token
load_dotenv()

# Ensure HuggingFace API key is loaded from .env
if not os.getenv("HUGGINGFACE_API_TOKEN"):
    # Try to load from .env file explicitly
    from pathlib import Path
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)
    
    # If still not found, show warning but continue (HF has free tier)
    if not os.getenv("HUGGINGFACE_API_TOKEN"):
        st.warning("💡 Consider adding HUGGINGFACE_API_TOKEN to .env for better performance")

class CompanyDocumentManager:
    def __init__(self, company_name="Company"):
        self.company_name = company_name
        self.storage_path = f"company_docs_{company_name.lower().replace(' ', '_')}"
        self.vector_store_path = f"{self.storage_path}/vector_store"
        self.metadata_path = f"{self.storage_path}/metadata.json"
        
        # Create storage directory
        os.makedirs(self.storage_path, exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings - Always use HuggingFace (free and reliable)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}  # Better similarity search
        )
    
    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """Load document based on file type"""
        try:
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_type == 'txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_type == 'csv':
                loader = CSVLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            return documents
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return []
    
    def process_and_store_documents(self, uploaded_files):
        """Process uploaded files and update the company knowledge base"""
        all_documents = []
        processed_files = []
        
        # Load existing metadata
        metadata = self.load_metadata()
        
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Determine file type
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Load document
            documents = self.load_document(tmp_file_path, file_type)
            
            if documents:
                # Add metadata to each document
                for doc in documents:
                    doc.metadata.update({
                        'source_file': uploaded_file.name,
                        'file_type': file_type,
                        'upload_date': datetime.now().isoformat(),
                        'company': self.company_name
                    })
                
                # Split documents
                split_docs = self.text_splitter.split_documents(documents)
                all_documents.extend(split_docs)
                
                processed_files.append({
                    'filename': uploaded_file.name,
                    'file_type': file_type,
                    'chunks': len(split_docs),
                    'upload_date': datetime.now().isoformat()
                })
            
            # Clean up temp file
            os.unlink(tmp_file_path)
        
        if all_documents:
            # Create or update vector store
            if os.path.exists(self.vector_store_path):
                # Load existing vector store and add new documents
                try:
                    vector_store = FAISS.load_local(
                        self.vector_store_path, 
                        self.embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    vector_store.add_documents(all_documents)
                except Exception as e:
                    # If loading fails, create new vector store
                    st.warning(f"Creating new vector store: {str(e)[:100]}")
                    vector_store = FAISS.from_documents(all_documents, self.embeddings)
            else:
                # Create new vector store
                vector_store = FAISS.from_documents(all_documents, self.embeddings)
            
            # Save vector store
            vector_store.save_local(self.vector_store_path)
            
            # Update metadata
            metadata['files'].extend(processed_files)
            metadata['last_update'] = datetime.now().isoformat()
            metadata['total_files'] = len(metadata['files'])
            
            self.save_metadata(metadata)
            
            return len(all_documents), processed_files
        
        return 0, []
    
    def load_metadata(self):
        """Load company document metadata"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'company_name': self.company_name,
                'files': [],
                'created_date': datetime.now().isoformat(),
                'last_update': None,
                'total_files': 0
            }
    
    def save_metadata(self, metadata):
        """Save company document metadata"""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_vector_store(self):
        """Load the company's vector store"""
        if os.path.exists(self.vector_store_path):
            try:
                return FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                st.error(f"Error loading vector store: {str(e)}")
                return None
        return None

class CompanyAssistant:
    def __init__(self, company_name="Company"):
        self.company_name = company_name
        self.doc_manager = CompanyDocumentManager(company_name)
        
        # Initialize LLM - Use Hugging Face Hub with your API token
        self.llm = self._initialize_huggingface_llm()
        
        # Custom prompt template optimized for Hugging Face models
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "company_name"],
            template="""Use the company documents to answer the question accurately.

Company: {company_name}

Documents: {context}

Question: {question}

Answer based only on the company documents. If the answer is not in the documents, say "I don't have that information in the company documents."

Answer:"""
        )
    
    def _initialize_huggingface_llm(self):
        """Initialize Hugging Face LLM with fallback options"""
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        # List of models to try (in order of preference)
        models_to_try = [
            "microsoft/DialoGPT-large",
            "google/flan-t5-large", 
            "google/flan-t5-base",
            "facebook/blenderbot-400M-distill"
        ]
        
        for model_name in models_to_try:
            try:
                llm = HuggingFaceHub(
                    repo_id=model_name,
                    model_kwargs={
                        "temperature": 0.1,
                        "max_new_tokens": 512,
                        "do_sample": True,
                        "top_p": 0.9,
                        "top_k": 50
                    },
                    huggingfacehub_api_token=hf_token
                )
                
                # Test the model with a simple query
                test_response = llm("Test question: What is 2+2?")
                if test_response:
                    st.success(f"✅ Using Hugging Face model: {model_name}")
                    return llm
                    
            except Exception as e:
                st.warning(f"⚠️ Model {model_name} failed: {str(e)[:100]}...")
                continue
        
        # If all models fail
        st.error("❌ Unable to initialize any Hugging Face model. Please check your internet connection and API token.")
        return None
    
    def setup_qa_chain(self):
        """Setup the QA chain with company documents"""
        vector_store = self.doc_manager.get_vector_store()
        
        if not vector_store:
            return None
        
        if not self.llm:
            return None
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt_template
            }
        )
        
        return qa_chain
    
    def answer_question(self, question: str):
        """Answer user question using company documents"""
        qa_chain = self.setup_qa_chain()
        
        if not qa_chain:
            return {
                'answer': "I'm not ready to help yet. Please make sure documents are uploaded and there's a stable internet connection for Hugging Face API.",
                'sources': []
            }
        
        try:
            result = qa_chain({
                "query": question,
                "company_name": self.company_name
            })
            
            return {
                'answer': result['result'],
                'sources': result.get('source_documents', [])
            }
        except Exception as e:
            return {
                'answer': f"I encountered an error while searching for information: {str(e)}",
                'sources': []
            }

def company_admin_interface():
    """Interface for company to manage documents"""
    st.header("🏢 Company Document Management")
    
    # Company name input
    company_name = st.text_input("Enter Your Company Name", value="", placeholder="e.g., TechFlow Solutions")
    
    if not company_name.strip():
        st.info("👆 Please enter your company name to get started")
        return
    
    # Initialize document manager for this company
    if 'doc_manager' not in st.session_state or st.session_state.get('admin_company_name') != company_name:
        st.session_state.doc_manager = CompanyDocumentManager(company_name)
        st.session_state.admin_company_name = company_name
    
    st.success(f"📋 Managing documents for: **{company_name}**")
    
    # Display current status
    metadata = st.session_state.doc_manager.load_metadata()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", metadata['total_files'])
    with col2:
        if metadata['last_update']:
            last_update = datetime.fromisoformat(metadata['last_update']).strftime("%Y-%m-%d %H:%M")
            st.metric("Last Updated", last_update)
        else:
            st.metric("Last Updated", "Never")
    with col3:
        vector_store = st.session_state.doc_manager.get_vector_store()
        if vector_store:
            st.metric("Knowledge Base", "✅ Active")
        else:
            st.metric("Knowledge Base", "❌ Empty")
    
    # File upload section
    st.subheader("📁 Upload Company Documents")
    st.markdown(f"Upload documents containing **{company_name}** information, policies, FAQs, procedures, etc.")
    
    uploaded_files = st.file_uploader(
        "Choose company documents",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'csv'],
        help="Upload PDF, TXT, or CSV files containing company information"
    )
    
    if uploaded_files and st.button("📤 Upload & Process Documents", type="primary"):
        with st.spinner(f"Processing documents for {company_name}..."):
            total_chunks, processed_files = st.session_state.doc_manager.process_and_store_documents(uploaded_files)
            
            if total_chunks > 0:
                st.success(f"✅ Successfully processed {len(processed_files)} files into {total_chunks} knowledge chunks!")
                
                for file_info in processed_files:
                    st.info(f"📄 {file_info['filename']}: {file_info['chunks']} chunks")
                
                st.balloons()
                st.markdown("🎉 **Your company documents are now ready for user queries!**")
            else:
                st.error("❌ No documents were processed successfully.")
    
    # Current documents display
    if metadata['files']:
        st.subheader("📚 Current Knowledge Base")
        for file_info in metadata['files']:
            with st.expander(f"📄 {file_info['filename']}"):
                st.write(f"**File Type:** {file_info['file_type'].upper()}")
                st.write(f"**Chunks Created:** {file_info['chunks']}")
                st.write(f"**Upload Date:** {file_info['upload_date']}")
    else:
        st.info("📭 No documents uploaded yet. Upload your first company document above!")

def user_chat_interface():
    """Interface for users to chat with company assistant"""
    st.header("💬 Ask Questions About Any Company")
    
    # User enters company name they want to query
    st.markdown("Enter the company name you'd like to ask questions about:")
    
    user_company_name = st.text_input(
        "Company Name", 
        value="", 
        placeholder="e.g., TechFlow Solutions",
        key="user_company_input"
    )
    
    if not user_company_name.strip():
        st.info("👆 Please enter a company name to get started")
        st.markdown("""
        ### 💡 How it works:
        1. **Enter company name** you want to ask about
        2. **System checks** if documents exist for that company
        3. **Ask questions** and get answers from their documents
        4. **View sources** to see which documents provided the answer
        """)
        return
    
    # Check if company documents exist
    temp_doc_manager = CompanyDocumentManager(user_company_name)
    vector_store = temp_doc_manager.get_vector_store()
    metadata = temp_doc_manager.load_metadata()
    
    if not vector_store or metadata['total_files'] == 0:
        st.error(f"📭 **No documents found for '{user_company_name}'**")
        st.markdown(f"""
        **Sorry!** There are no uploaded documents for **{user_company_name}**.
        
        **Possible solutions:**
        - Check if the company name is spelled correctly
        - Ask the company to upload their documents in Admin mode
        - Try a different company name
        
        **Available companies with documents:**
        """)
        
        # Show available companies
        try:
            import glob
            company_dirs = glob.glob("company_docs_*")
            available_companies = []
            
            for dir_path in company_dirs:
                company_folder_name = dir_path.replace("company_docs_", "").replace("_", " ")
                temp_mgr = CompanyDocumentManager(company_folder_name)
                temp_meta = temp_mgr.load_metadata()
                if temp_meta['total_files'] > 0:
                    available_companies.append(f"• **{temp_meta['company_name']}** ({temp_meta['total_files']} documents)")
            
            if available_companies:
                for company in available_companies:
                    st.markdown(company)
            else:
                st.markdown("• *No companies have uploaded documents yet*")
                
        except Exception as e:
            st.markdown("• *Unable to scan for available companies*")
        
        return
    
    # Company documents found - show success and stats
    st.success(f"✅ **Found documents for '{user_company_name}'**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", metadata['total_files'])
    with col2:
        if metadata['last_update']:
            last_update = datetime.fromisoformat(metadata['last_update']).strftime("%Y-%m-%d")
            st.metric("🔄 Last Updated", last_update)
    with col3:
        st.metric("🤖 AI Status", "Ready")
    
    st.markdown(f"**Ask me anything about {user_company_name}!** I'll search through their documents to help you.")
    
    # Initialize assistant for this company
    if 'user_assistant' not in st.session_state or st.session_state.get('user_company_name') != user_company_name:
        st.session_state.user_assistant = CompanyAssistant(user_company_name)
        st.session_state.user_company_name = user_company_name
        st.session_state.chat_messages = []  # Reset chat for new company
    
    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander(f"📚 Sources from {user_company_name} documents"):
                        for i, source in enumerate(message["sources"]):
                            st.write(f"**Source {i+1}:** {source.metadata.get('source_file', 'Unknown file')}")
                            st.write(source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content)
                            if i < len(message["sources"]) - 1:
                                st.write("---")
    
    # Chat input
    if prompt := st.chat_input(f"Ask {user_company_name} anything..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"Searching {user_company_name} documents..."):
                result = st.session_state.user_assistant.answer_question(prompt)
                
                st.write(result['answer'])
                
                # Add assistant message
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": result['answer'],
                    "sources": result['sources']
                })
                
                # Show sources
                if result['sources']:
                    with st.expander(f"📚 Sources from {user_company_name} documents"):
                        for i, source in enumerate(result['sources']):
                            st.write(f"**Source {i+1}:** {source.metadata.get('source_file', 'Unknown file')}")
                            st.write(source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content)
                            if i < len(result['sources']) - 1:
                                st.write("---")

def main():
    st.set_page_config(
        page_title="Company Knowledge Assistant",
        page_icon="🏢",
        layout="wide"
    )
    
    st.title("🤖 Company Knowledge Assistant")
    st.markdown("**Powered by Hugging Face - Free RAG chatbot for company documentation**")
    
    # Show current model status
    if 'assistant' in st.session_state:
        if st.session_state.assistant.llm:
            st.success("🟢 Hugging Face AI Model Active")
        else:
            st.error("🔴 AI Model Not Available")
    
    # Mode selection
    st.sidebar.header("🎯 Select Mode")
    mode = st.sidebar.radio(
        "Choose your role:",
        ["👥 User (Ask Questions)", "👨‍💼 Admin (Manage Documents)"]
    )
    
    # Instructions
    with st.sidebar.expander("🤖 Hugging Face Setup"):
        st.markdown("""
        ### ✅ Free Hugging Face AI!
        - **Models**: Multiple fallback options
        - **Cost**: Completely FREE
        - **Token**: Optional (get from HF settings)
        
        ### Current Features:
        - 📄 Document processing (PDF, TXT, CSV)
        - 🔍 Semantic search with embeddings
        - 🤖 AI responses via Hugging Face
        - 📚 Source attribution
        
        ### For Company Admins:
        1. Set your company name
        2. Upload company documents 
        3. Documents processed into knowledge base
        
        ### For Users:
        1. Ask questions about the company
        2. Get AI answers from company docs
        3. View sources used for each answer
        
        ### Example Questions:
        - "What services do you offer?"
        - "What's your refund policy?"
        - "How do I contact support?"
        - "What are your business hours?"
        
        ### Hugging Face Models Used:
        - **Embeddings**: all-MiniLM-L6-v2
        - **LLM**: DialoGPT/Flan-T5 (auto-selected)
        """)
    
    if mode == "👨‍💼 Admin (Manage Documents)":
        company_admin_interface()
    else:
        user_chat_interface()

if __name__ == "__main__":
    main()