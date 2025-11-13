import os
from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableMap, RunnablePassthrough
from src.components.vectorstore import FaissVectorStore
from src.components.local_llm import LoadLLM

load_dotenv()


class RAGSearch:
    """
    LCEL-based RAG pipeline (recommended for LangChain 1.x).
    Retrieves company-relevant information & answers strictly
    sales/marketing related questions using a local LLM.
    """

    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2"):
        
        # Initialize FAISS retriever
        self.vectorstore = FaissVectorStore(persist_dir,embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir,"faiss.index")
        meta_path = os.path.join(persist_dir,"metadata.pkl")
        if not(os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        loader = LoadLLM(model_name="phi")
        self.llm = loader.get_model()

    # def _build_rag_chain(self):
    #     """
    #     Build LCEL Retrieval-Augmented Generation pipeline.
    #     """
    #     llm_loader = LoadLLM(model_name="phi")
    #     llm = llm_loader.get_model()



    #     prompt = PromptTemplate(
    #         template=prompt_template,
    #         input_variables=["context", "question"]
    #     )

    #     # LCEL RAG pipeline
    #     rag_chain = (
    #         RunnableMap({
    #             "context": lambda x: self.retriever.get_relevant_documents(x["question"]),
    #             "question": lambda x: x["question"]
    #         })
    #         | prompt
    #         | llm
    #         | RunnablePassthrough()
    #     )

    #     print("[INFO] 🔗 LCEL RAG pipeline successfully built.")
    #     return rag_chain

    def search_and_summarize(self, query: str,top_k:int = 5) -> str:
        results = self.vectorstore.query(query,top_k=top_k)
        texts = [r["metadata"].get("text", "")for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant document found."
        prompt = """
        You are the official AI assistant for [COMPANY NAME].

        ONLY answer questions about:
        - Sales
        - Marketing
        - Lead generation
        - Company products & services
        - Company policies

        If a user asks anything outside these domains, reply:
        "I can only assist with sales and marketing questions related to our company."

        Do NOT mention:
        - Other companies
        - Competitors
        - External brands
        - Unrelated topics

        If someone asks about another company:
        "I cannot discuss other companies. Please ask about our company's sales and marketing."

        Use the context (if provided) to answer clearly.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        response = self.llm.invoke([prompt])
        return response.content
    

# Example Usage
if __name__=="__main__":
    rag_search = RAGSearch()
    query = "What is sales?"
    summary = rag_search.search_and_summarize(query,top_k=3)
    print("Summary:", summary)