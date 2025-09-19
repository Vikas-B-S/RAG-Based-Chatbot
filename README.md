# 📚 RAG Based Chatbot  

A **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit**, **LangChain**, **FAISS**, and **Google Gemini**.  
This chatbot allows you to upload a **PDF document** and then ask questions about its content. It retrieves the most relevant chunks of text from the document and generates **concise, context-aware answers** using Gemini.  

---

## 🚀 Features
- 📂 Upload any PDF document.  
- 🔍 Automatic text chunking & semantic search using FAISS.  
- 💬 Interactive chatbot with **chat history**.  
- 🧠 Retrieval-Augmented answers (no hallucinations).  
- 🎨 Clean Streamlit UI with chat-style messages.  

---

## 🛠️ Tech Stack
- **Python**  
- **Streamlit** (UI)  
- **LangChain** (Text splitting + FAISS Retriever)  
- **FAISS** (Vector database)  
- **HuggingFace Embeddings**  
- **Google Gemini API**  

---

## ⚙️ Setup Instructions
1. Clone the repository:
   ```bash
   git https://github.com/Vikas-B-S/RAG_Based_Chatbot.git
   cd RAG_Based_Chatbot
   ```

2. Create a virtual environment & install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your **Google API Key**:
   ```bash
   export GOOGLE_API_KEY="your_api_key"
   ```
   (For Windows PowerShell)  
   ```powershell
   $env:GOOGLE_API_KEY="your_api_key"
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📖 Usage
1. Upload a **PDF file** from the sidebar.  
2. Ask questions about the document in the chat box.  
3. The chatbot retrieves the most relevant text and generates an answer.  
4. Continue the conversation with context-aware follow-ups.  


