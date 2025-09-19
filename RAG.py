import os
import google.generativeai as genai
from langchain.vectorstores import FAISS  # This will be the vector database
from langchain_community.embeddings import HuggingFaceEmbeddings # To perform word embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # This for chunking
from pypdf import PdfReader
import faiss
import streamlit as st
from pdfextracter import text_extractor_pdf

# Create the main page
st.title(':green[RAG Based CHATBOT]')
tips = """
### How to Use This RAG Chatbot  

Welcome! Follow these steps to get started:  

1. **Upload a PDF document** from the sidebar (only `.pdf` files are supported).  
2. The document will be **split into chunks** and stored in a smart search database.  
3. Type your **question** in the chat box below and click **Send**.  
4. The chatbot will **retrieve the most relevant parts** of your document and generate a helpful answer.  
5. Your chat history is saved, so you can continue asking follow-up questions.  

---

**Tips for Best Results**  
- Keep your questions **specific and clear**.  
- If the chatbot gives a vague answer, try **rephrasing your query**.  
- Upload a **well-formatted PDF** for more accurate responses.  
- You can re-upload a new PDF anytime to start fresh.  

Enjoy exploring your documents with AI assistance! 🚀  
"""
st.markdown(tips)


# Load PDF in Side Bar
st.sidebar.title(':orange[UPLOAD YOUR DOCUMENT HERE (PDF Only)]')
file_uploaded = st.sidebar.file_uploader('Upload File')
if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)
    # Step 1: Configure the models

    # Configure LLM
    key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Configure Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Step 2 : Chunking (Create Chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap = 200)
    chunks = splitter.split_text(file_text)

    # Step 3: Create FAISS Vector Store
    vector_store = FAISS.from_texts(chunks,embedding_model)

    # Step 4: Configure retriever
    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    # Lets create a function that takes query and return the generated text
    def generate_response(query):
        # Step 6 : Retrieval (R)
        retrived_docs = retriever.get_relevant_documents(query=query)
        context = ' '.join([doc.page_content for doc in retrived_docs])

        # Step 7: Write a Augmeneted prompt (A)
        prompt = f"""
        [System Role]
        You are a Retrieval-Augmented Generation (RAG) assistant.
        Your job is to help the user by answering questions **only from the given context**.
        If the context partially covers the query, answer with what is available and clearly note that it's partial.
        If the answer is not in the context, clearly say:
        "I could not find relevant information in the uploaded document."

        ---------------------
        Context: {context}
        ---------------------

        User Query: {query}

        Instructions:
        - Use the context above to answer.
        - Keep responses clear, structured, and concise.
        - Never invent facts outside the provided document.
        - If relevant, summarize rather than copy long passages.
        """


        # Step 9: Generation (G)
        content = llm_model.generate_content(prompt)
        return content.text

    
    # Lets create a chatbot in order to start the converstaion
    # Initialize chat if there is no history
    if 'history' not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            if role == "user":
                st.markdown(f"**👤 User:** {msg['text']}")
            else:
                st.markdown(f"**🤖 Assistant:** {msg['text']}")

    # Input from the user (Using Steamlit Form)
    # Chat input (fixed at bottom)
    if user_input := st.chat_input("Enter your query here..."):
        # Save user message
        st.session_state.history.append({"role": "user", "text": user_input})

        # Generate chatbot response
        model_output = generate_response(user_input)

        # Save chatbot reply
        st.session_state.history.append({"role": "assistant", "text": model_output})

        st.rerun()


        