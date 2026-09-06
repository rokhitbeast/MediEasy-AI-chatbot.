import os

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Import our refactored pipeline function
from create_memory_for_llm import build_vector_store

# Load environment variables
load_dotenv()
DB_FAISS_PATH="vectorstore/db_faiss" 
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "Qwen/Qwen2.5-1.5B-Instruct"

# Our optimized prompt
custom_prompt_template = """You are MediBot, a helpful medical assistant.
Use the provided context to answer the user's question.
If the context contains relevant information, summarize it to provide a clear answer.
If the context does not contain any relevant information to answer the question, reply exactly: I don't know.
Do not use outside general knowledge.

Context:
{context}

Question:
{question}

Answer concisely:"""


@st.cache_resource
def vectorstore():
    if not os.path.exists(DB_FAISS_PATH):
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db


def set_custom_prompt():
    prompt=PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt


def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        provider="featherless-ai",
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=endpoint)


def check_password():
    """Returns `True` if the user had the correct password."""
    env_user = os.environ.get("MEDIBOT_USERNAME")
    env_pass = os.environ.get("MEDIBOT_PASSWORD")
    
    if not env_user or not env_pass:
        st.error("⚠️ Authentication not configured. Please set MEDIBOT_USERNAME and MEDIBOT_PASSWORD in your .env file.")
        st.stop()

    def password_entered():
        if (st.session_state["username"] == env_user and
            st.session_state["password"] == env_pass):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # --- UI POLISH: Centered Login Card ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True) # Spacer
        st.title("⚕️ MediBot")
        st.caption("Secure Portal Login")
        
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        
        # Make button stretch across the column
        st.button("Login", on_click=password_entered, type="primary", use_container_width=True)

        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("😕 Username or password incorrect")
    
    return False


def main():
    # --- UI POLISH: Page Configuration ---
    # MUST be the first Streamlit command run
    st.set_page_config(page_title="MediBot | RAG Assistant", page_icon="⚕️", layout="wide")

    # --- AUTHENTICATION CHECK ---
    if not check_password():
        st.stop()
        
    # --- UI POLISH: Enhanced Sidebar ---
    with st.sidebar:
        st.title("⚕️ MediBot Settings")
        st.divider()
        
        st.markdown("### 📚 Knowledge Base")
        uploaded_files = st.file_uploader("Upload Medical Documents (PDF)", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Process Documents", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
                    os.makedirs("Data", exist_ok=True)
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("Data", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    success = build_vector_store()
                    if success:
                        st.cache_resource.clear()
                        st.success(f"Added {len(uploaded_files)} document(s)!")
                    else:
                        st.error("Failed to process documents.")
        
        # Display currently loaded files
        st.markdown("#### Currently Loaded:")
        if os.path.exists("Data") and os.listdir("Data"):
            for f in os.listdir("Data"):
                if f.endswith(".pdf"):
                    st.caption(f"📄 {f}")
        else:
            st.caption("No documents loaded yet.")
            
        st.divider()
        st.markdown("### 👤 Account")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # --- UI POLISH: Main Chat Interface ---
    st.title("⚕️ MediBot Clinical Assistant")
    
    # Critical for health-related projects
    st.warning("**Medical Disclaimer:** MediBot is an AI assistant designed for educational and informational purposes based on uploaded documents. It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.", icon="⚠️")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! I'm MediBot. Upload some medical documents in the sidebar, and ask me any questions about them!"}]
        
    for message in st.session_state.messages:
        # Use custom avatars
        avatar = "🧑‍⚕️" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):   
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a medical question based on the knowledge base...")
    if prompt:
        st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            db = vectorstore()
            if db is None:
                st.error("Knowledge base is empty. Please upload a medical PDF from the sidebar first.", icon="📂")
                st.stop()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )
            
            with st.spinner("Analyzing documents..."):
                response = qa_chain.invoke({"query": prompt})

            result = response['result']
            source_documents = response['source_documents']

            with st.chat_message("assistant", avatar="🧑‍⚕️"):
                st.markdown(result)
                
                # --- UI POLISH: Cleaner Sources Display ---
                with st.expander(f"📚 View Sources ({len(source_documents)} chunks retrieved)"):
                    for index, document in enumerate(source_documents, start=1):
                        metadata = document.metadata
                        source = metadata.get("source", "Unknown source")
                        filename = os.path.basename(source) 
                        page = metadata.get("page")
                        page_label = f" | Page {page + 1}" if isinstance(page, int) else ""
                        
                        st.markdown(f"**{index}. {filename}** {page_label}")
                        # Wrap the raw text in an info box for a cleaner look
                        st.info(document.page_content, icon="🔍")
                        
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()