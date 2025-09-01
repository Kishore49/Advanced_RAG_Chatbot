# Advanced RAG Chatbot with Streamlit 


## ⚙️ How It Works

1. Upload a PDF file.
2. The text is extracted and split into smaller chunks.
3. The chunks are stored in a **FAISS vector database** for fast semantic search.
4. When you ask a question, the most relevant chunks are retrieved.
5. A Hugging Face model (**Flan-T5**) generates the final answer using those chunks.

---

## 🛠️ Tech Stack

* Streamlit
* PyPDF2
* LangChain
* FAISS
* Hugging Face

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Kishore49/Advanced_RAG_Chatbot.git
cd advanced-rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```
