from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader,CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir:str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to Langchain document structure
    Supported: PDF, TXT, CSV, Excel, Word, Json
    """

    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    #PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load PDF {pdf_file}: {e}")


    # TEXT files (.txt, .md)
    text_files = list(data_path.glob('**/*.txt')) + list(data_path.glob('**/*.md'))
    print(f"[DEBUG] Found {len(text_files)} text files: {[str(f) for f in text_files]}")
    for text_file in text_files:
        print(f"[DEBUG] Loading text: {text_file}")
        try:
            loader = TextLoader(str(text_file), encoding='utf-8')
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} text docs from {text_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load text {text_file}: {e}")


    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load CSV {csv_file}: {e}")


    # JSON files# JSON files
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema=".[]",
                text_content=False
                )
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load JSON {json_file}: {e}")


    # Word (.docx) files
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_files)} DOCX files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        print(f"[DEBUG] Loading DOCX: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} DOCX docs from {docx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load DOCX {docx_file}: {e}")

            
    # Excel (.xls, .xlsx) files
    excel_files = list(data_path.glob('**/*.xls')) + list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(excel_files)} Excel files: {[str(f) for f in excel_files]}")
    for excel_file in excel_files:
        print(f"[DEBUG] Loading Excel: {excel_file}")
        try:
            loader = UnstructuredExcelLoader(str(excel_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {excel_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load Excel {excel_file}: {e}")

    return documents