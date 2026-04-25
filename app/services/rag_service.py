import os
import threading
from typing import List, Dict, Any, Optional, TypedDict, Literal
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "../../data/knowledge_base")
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/faiss_index_langchain")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Define Graph State
class GraphState(TypedDict):
    question: str          # Câu hỏi gốc của người dùng
    search_query: str     # Câu hỏi đã được tối ưu hóa để tìm kiếm
    documents: List[Any]   # Danh sách tài liệu tìm được
    answer: str            # Câu trả lời cuối cùng
    sources: List[str]     # Nguồn trích dẫn
    iteration_count: int   # Số lần thử lại (tránh vòng lặp vô tận)

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
        
        self.vector_store = None
        self.app = None
        
        if GOOGLE_API_KEY:
            print(f"[RAG] 🔑 Initializing Advanced LangGraph (Self-Reflective)...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3 # Tăng một chút để văn phong tự nhiên hơn
            )
            self.initialize()
        else:
            print("[RAG] ⚠️ GOOGLE_API_KEY missing. RAG service will not be available.")

    def initialize(self):
        try:
            if os.path.exists(VECTOR_DB_PATH):
                self.vector_store = FAISS.load_local(VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
                print(f"[RAG] 🟢 Loaded FAISS index.")
            else:
                self.rebuild_index()
            
            self._build_graph()
            self._initialized = True
            print("[RAG] ✅ Advanced LangGraph Service Ready!")
            
        except Exception as e:
            print(f"[RAG] ❌ Initialization failed: {e}")

    def rebuild_index(self):
        """Standard FAISS indexing logic"""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
            return

        loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        docs = loader.load()
        if not docs: return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.vector_store.save_local(VECTOR_DB_PATH)
        print(f"[RAG] ✅ Index Rebuilt.")

    def _build_graph(self):
        """Builds the Self-Reflective Graph"""
        workflow = StateGraph(GraphState)

        # Add Nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("transform_query", self._transform_query_node)

        # Entry Point
        workflow.set_entry_point("retrieve")

        # Edges
        workflow.add_edge("retrieve", "grade_documents")
        
        # Rẽ nhánh dựa trên kết quả chấm điểm tài liệu
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "generate": "generate",
                "transform_query": "transform_query"
            }
        )
        
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)

        self.app = workflow.compile()

    # --- Graph Logic Nodes ---

    def _retrieve_node(self, state: GraphState):
        query = state.get("search_query") or state["question"]
        print(f"[Graph] 🔍 Retrieving for: {query}")
        try:
            docs = self.vector_store.similarity_search(query, k=3)
        except Exception as e:
            print(f"[Graph] ⚠️ Retrieval error (possibly Embedding API 500): {e}")
            docs = [] # Trả về list rỗng để AI tự trả lời bằng kiến thức của nó nếu RAG lỗi
            
        return {"documents": docs, "iteration_count": state.get("iteration_count", 0) + 1}

    def _grade_documents_node(self, state: GraphState):
        """Chấm điểm hàng loạt tất cả tài liệu để tiết kiệm API Quota"""
        if not state["documents"]:
            return {"documents": []}
            
        print(f"[Graph] ⚖️ Batch Grading {len(state['documents'])} documents...")
        
        # Gộp các tài liệu lại để chấm điểm 1 lần duy nhất
        context_str = ""
        for i, doc in enumerate(state["documents"]):
            context_str += f"[Tài liệu ID: {i}]\n{doc.page_content}\n---\n"

        prompt = PromptTemplate.from_template(
            """Bạn là chuyên gia thẩm định dữ liệu y khoa. Hãy xem xét các tài liệu dưới đây và xác định xem cái nào thực sự có chứa thông tin hữu ích để trả lời câu hỏi của người dùng.
            
            Trả về một danh sách các ID của những tài liệu liên quan.
            Định dạng trả về duy nhất là JSON: {{"relevant_ids": [0, 2]}}
            Nếu không có cái nào liên quan, trả về: {{"relevant_ids": []}}

            Câu hỏi: {question}
            Danh sách tài liệu:
            {context_str}
            """
        )
        
        grader = prompt | self.llm | JsonOutputParser()
        
        try:
            print("[API CALL] 🚀 Gửi yêu cầu THẨM ĐỊNH tài liệu tới Gemini...")
            res = grader.invoke({"question": state["question"], "context_str": context_str})
            relevant_ids = res.get("relevant_ids", [])
            relevant_docs = [state["documents"][i] for i in relevant_ids if int(i) < len(state["documents"])]
            print(f"[Graph] ✅ Lọc được {len(relevant_docs)}/{len(state['documents'])} tài liệu liên quan.")
            return {"documents": relevant_docs}
        except Exception as e:
            print(f"[Graph] ⚠️ Lỗi chấm điểm hàng loạt: {e}. Giữ nguyên toàn bộ tài liệu.")
            return {"documents": state["documents"]}

    def _decide_to_generate(self, state: GraphState) -> Literal["generate", "transform_query"]:
        """Hàm điều hướng: Sinh câu trả lời hay sửa câu hỏi?"""
        # Giảm xuống tối đa 2 lần thử để tiết kiệm Quota
        if not state["documents"] and state["iteration_count"] < 2:
            print("[Graph] 🔄 Không tìm thấy tài liệu phù hợp. Chuyển sang sửa câu hỏi (Lần 1).")
            return "transform_query"
        return "generate"

    def _transform_query_node(self, state: GraphState):
        """Viết lại câu hỏi để tìm kiếm tốt hơn"""
        print("[Graph] 📝 Transforming query for better retrieval...")
        
        prompt = PromptTemplate.from_template(
            """Bạn là chuyên gia về tối ưu hóa tìm kiếm y khoa. 
            Hãy viết lại câu hỏi sau đây để có kết quả tìm kiếm tốt hơn trong cơ sở dữ liệu da liễu.
            Chỉ trả về câu hỏi mới, không giải thích gì thêm.
            
            Câu hỏi gốc: {question}
            """
        )
        
        rewriter = prompt | self.llm | StrOutputParser()
        print("[API CALL] 🚀 Gửi yêu cầu VIẾT LẠI CÂU HỎI tới Gemini...")
        new_query = rewriter.invoke({"question": state["question"]})
        return {"search_query": new_query}

    def _generate_node(self, state: GraphState):
        """Sinh câu trả lời cuối cùng với phong cách bác sĩ tế nhị"""
        print("[Graph] ✨ Generating final response...")
        
        # Nếu không có tài liệu nào liên quan sau các lượt tìm kiếm
        if not state["documents"]:
            # Gọi AI kiểm tra xem câu hỏi có thuộc chuyên môn da liễu không
            check_prompt = PromptTemplate.from_template(
                "Câu hỏi này có liên quan đến bệnh da liễu, tóc, móng hoặc thẩm mỹ da không? Trả lời 'yes' hoặc 'no'.\nCâu hỏi: {question}"
            )
            checker = check_prompt | self.llm | StrOutputParser()
            try:
                is_dermatology = checker.invoke({"question": state["question"]})
            except:
                is_dermatology = "yes" # Mặc định là yes nếu lỗi để trả lời an toàn

            if "yes" in is_dermatology.lower():
                return {
                    "answer": "DermCare rất tiếc khi chưa tìm thấy thông tin trùng khớp hoàn toàn với tình trạng bạn vừa mô tả trong kho dữ liệu chuyên môn hiện tại. Để đảm bảo an toàn và có chẩn đoán chính xác nhất, bạn có thể mô tả kỹ hơn triệu chứng hoặc đặt lịch thăm khám trực tuyến để bác sĩ chuyên khoa của chúng tôi hỗ trợ trực tiếp nhé!",
                    "sources": []
                }
            else:
                return {
                    "answer": "Dạ, hiện tại DermCare là trợ lý chuyên biệt hỗ trợ các vấn đề về Da liễu, Tóc và Móng thôi ạ. Câu hỏi này có vẻ nằm ngoài phạm vi chuyên môn của em mất rồi. Bạn có câu hỏi nào về tình trạng da, mụn hay các vấn đề thẩm mỹ da không? Em luôn sẵn sàng hỗ trợ bạn!",
                    "sources": []
                }

        context = "\n---\n".join([d.page_content for d in state["documents"]])
        sources = list(set([os.path.basename(d.metadata.get("source", "Unknown")) for d in state["documents"]]))
        
        prompt = PromptTemplate.from_template(
            """Bạn là bác sĩ chuyên khoa da liễu tại DermCare AI. 
            Dựa trên ngữ cảnh y khoa, hãy tư vấn cho bệnh nhân một cách CHUYÊN NGHIỆP và SÚC TÍCH.

            Quy tắc trình bày:
            1. Trả lời ngay vào trọng tâm câu hỏi.
            2. Sử dụng gạch đầu dòng cho các ý chính để dễ theo dõi trên điện thoại.
            3. Độ dài vừa phải (khoảng 2-3 đoạn ngắn), tránh viết quá dài như một bài báo.
            4. Luôn giữ thái độ ân cần và khuyên bệnh nhân thăm khám nếu cần thiết.
            5. Nếu không có trong tài liệu, hãy nói trung thực là bạn không rõ.

            Ngữ cảnh: {context}
            Câu hỏi: {question}

            Trả lời (tiếng Việt, ngắn gọn):"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        print("[API CALL] 🚀 Gửi yêu cầu SINH CÂU TRẢ LỜI tới Gemini...")
        answer = chain.invoke({"context": context, "question": state["question"]})
        
        return {"answer": answer, "sources": sources}

    async def query(self, question: str) -> dict:
        if not self._initialized: return {"answer": "Hệ thống đang khởi động, bạn vui lòng chờ giây lát nhé!", "sources": []}
        try:
            result = self.app.invoke({"question": question, "iteration_count": 0})
            return {
                "answer": result["answer"],
                "sources": result["sources"]
            }
        except Exception as e:
            # In lỗi thực tế ra terminal để lập trình viên theo dõi
            print(f"[RAG] ❌ Graph error: {e}")
            # Trả về lời xin lỗi lịch sự cho người dùng
            return {
                "answer": "Rất tiếc, hệ thống đang bận một chút hoặc gặp sự cố kết nối. Bạn vui lòng thử lại sau giây lát nhé!",
                "sources": []
            }

rag_service = RAGService()
