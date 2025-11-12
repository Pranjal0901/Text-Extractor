import os 
from dotenv import load_dotenv
from src.components.vectorstore import FaissVectorStore

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir:str = "faiss_store", embedding_model:str = "all-MiniLm"):
        self.vectorstore = FaissVectorStore(persist_dir,embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir,"")        