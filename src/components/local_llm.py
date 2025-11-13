from langchain_ollama import OllamaLLM
from langchain_core.language_models.llms import LLM

class LoadLLM:
    """
    Wrapper class to load and manage a locally running Ollama LLM model for LangChain.

    Requirements:
    - Ollama installed and running locally (https://ollama.com)
    - Model pulled via: `ollama pull mistral`

    Example:
        llm_loader = LoadLLM(model_name="mistral")
        llm = llm_loader.get_model()
    """

    def __init__(self, model_name: str = "phi", temperature: float = 0.2, num_ctx: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.llm = None

        print(f"[INFO] Initializing local LLM '{self.model_name}'...")
        self._load_model()

    def _load_model(self):
        """
        Internal method to load the local Ollama model.
        """
        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                temperature=self.temperature,
                num_ctx=self.num_ctx
            )
            print(f"[INFO] Local LLM '{self.model_name}' loaded successfully via Ollama.")
        except Exception as e:
            print(f"[ERROR] Failed to connect to Ollama or load model '{self.model_name}': {e}")
            raise RuntimeError(
                f"Unable to initialize local LLM '{self.model_name}'. "
                "Ensure Ollama is running and the model is pulled."
            )

    def get_model(self) -> LLM:
        """
        Return the loaded LLM instance for use in LangChain pipelines.
        """
        if self.llm is None:
            print("[WARN] LLM not loaded, attempting to reload...")
            self._load_model()
        return self.llm
