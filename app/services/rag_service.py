import os
import threading
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import faiss

load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "../../data/knowledge_base")
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/faiss_index")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class RAGService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RAGService, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.index = None
        self.documents = []  # List of text chunks
        self.metadata = []   # Corresponding filenames/sources
        
        if GOOGLE_API_KEY:
            # Mask API Key for security but show enough to verify
            masked_key = f"{GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-5:]}" if len(GOOGLE_API_KEY) > 10 else "too short"
            print(f"[RAG] 🔑 GOOGLE_API_KEY detected: {masked_key}")
            genai.configure(api_key=GOOGLE_API_KEY)
            self.initialize()
        else:
            print("[RAG] ⚠️ GOOGLE_API_KEY missing in .env. RAG service will not be available.")

    def initialize(self):
        try:
            # Load index and documents if they exist
            docs_path = VECTOR_DB_PATH + "_docs.npy"
            meta_path = VECTOR_DB_PATH + "_meta.npy"
            index_file = VECTOR_DB_PATH + ".index"

            if os.path.exists(index_file) and os.path.exists(docs_path):
                print("[RAG] 🟢 Loading existing vector index...")
                self.index = faiss.read_index(index_file)
                self.documents = np.load(docs_path, allow_pickle=True).tolist()
                self.metadata = np.load(meta_path, allow_pickle=True).tolist()
                self._initialized = True
                print(f"[RAG] ✅ Loaded {len(self.documents)} chunks.")
            else:
                print("[RAG] 🟡 Creating new vector index from scratch...")
                self.rebuild_index()
                
            print("[RAG] ✅ Service initialized successfully!")
            
        except Exception as e:
            print(f"[RAG] ❌ Initialization failed: {e}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using Gemini API"""
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'], dtype='float32')

    def rebuild_index(self):
        """Builds or rebuilds the index from .txt files in KNOWLEDGE_BASE_DIR"""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
            print(f"[RAG] ⚠️ Directory {KNOWLEDGE_BASE_DIR} created but empty.")
            return

        all_texts = []
        all_meta = []
        
        print(f"[RAG] 📂 Loading documents from {KNOWLEDGE_BASE_DIR}...")
        for filename in os.listdir(KNOWLEDGE_BASE_DIR):
            if filename.endswith(".txt"):
                file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Simple chunking by paragraph or fixed length
                        chunks = [content[i:i+1000] for i in range(0, len(content), 900)]
                        for chunk in chunks:
                            all_texts.append(chunk)
                            all_meta.append(filename)
                except Exception as e:
                    print(f"[RAG] ❌ Failed to read {filename}: {e}")

        if not all_texts:
            print("[RAG] ❌ No documents found to index.")
            return

        print(f"[RAG] 🔢 Generating embeddings for {len(all_texts)} chunks...")
        embeddings = []
        for i, text in enumerate(all_texts):
            if i % 10 == 0: print(f"[RAG] Embedding progress: {i}/{len(all_texts)}")
            embeddings.append(self._get_embedding(text))
        
        emb_matrix = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = emb_matrix.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(emb_matrix)
        
        self.documents = all_texts
        self.metadata = all_meta
        
        # Save local
        os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
        faiss.write_index(self.index, VECTOR_DB_PATH + ".index")
        np.save(VECTOR_DB_PATH + "_docs.npy", np.array(self.documents, dtype=object))
        np.save(VECTOR_DB_PATH + "_meta.npy", np.array(self.metadata, dtype=object))
        
        self._initialized = True
        print(f"[RAG] ✅ Index saved. Total chunks: {len(self.documents)}")

    async def query(self, question: str) -> dict:
        print(f"[RAG] 🔍 Received query: {question}")
        if not self._initialized:
            return {"answer": "Hệ thống kiến thức chưa sẵn sàng. Vui lòng kiểm tra GOOGLE_API_KEY hoặc thêm dữ liệu vào data/knowledge_base.", "sources": []}
        
        try:
            # 1. Embed query
            query_emb = genai.embed_content(
                model="models/gemini-embedding-001",
                content=question,
                task_type="retrieval_query"
            )['embedding']
            query_emb = np.array([query_emb], dtype='float32')
            
            # 2. Search FAISS
            k = 3
            D, I = self.index.search(query_emb, k)
            
            # 3. Collect context
            context_chunks = []
            sources = []
            for idx in I[0]:
                if idx < len(self.documents):
                    context_chunks.append(self.documents[idx])
                    sources.append(self.metadata[idx])
            
            context_text = "\n---\n".join(context_chunks)
            
            # 4. Generate Answer using Gemini
            model = genai.GenerativeModel('models/gemini-flash-latest')
            prompt = f"""Bạn là một trợ lý y khoa chuyên về da liễu. 
Sử dụng thông tin sau đây để trả lời câu hỏi của người dùng một cách chi tiết và chuyên nghiệp nhất có thể. 
Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không biết, đừng cố bịa ra câu trả lời.

Ngữ cảnh:
{context_text}

Câu hỏi: {question}

Trả lời bằng tiếng Việt:"""
            
            response = model.generate_content(prompt)
            return {
                "answer": response.text,
                "sources": list(set(sources))
            }
            
        except Exception as e:
            print(f"[RAG] ❌ Query failed: {e}")
            return {"answer": f"Đã xảy ra lỗi khi truy vấn: {str(e)}", "sources": []}

rag_service = RAGService()
