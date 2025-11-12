from src.components.data_loader import load_all_documents
# from src.components.vectorstore import FaissVectorStore
# from src.components.search import RAGSearch
from src.components.embedding import EmbeddingPipeline
from src.components.vectorstore import FaissVectorStore
# Example Usage

if __name__=="__main__":
    #docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    print(store.query("Who is the founder of Tata?", top_k=3))