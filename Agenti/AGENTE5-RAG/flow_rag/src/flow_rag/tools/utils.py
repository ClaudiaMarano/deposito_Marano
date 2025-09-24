from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
import os 
from dataclasses import dataclass
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "faiss_index"
    # Text splitting
    chunk_size: int = 700
    chunk_overlap: int = 250
    # Retriever (MMR)
    search_type: str = "mmr"  # "mmr" o "similarity"
    k: int = 4  # risultati finali
    fetch_k: int = 20  # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3  # 0 = diversificazione massima, 1 = pertinenza massima
    endpoint = os.getenv("AZURE_API_BASE")
    subscription_key = os.getenv("AZURE_API_KEY")
    # Embedding Azure
    api_version = "2024-12-01-preview"
    model_name_emb = "text-embedding-ada-002"
    deployment_emb = "text-embedding-ada-002"
    # Azure
    model_name_chat = "gpt-4o"
    deployment_chat = "gpt-4o"


SETTINGS = Settings()
def get_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """
    Restituisce un client di Azure configurato.
    """
    return AzureOpenAIEmbeddings(
        model=settings.deployment_emb,
        api_version=settings.api_version,
        azure_endpoint=settings.endpoint,
        api_key=settings.subscription_key,
    )


def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Carica tutti i file di testo da una cartella come Documenti.
    """
    docs = []
    for file_path in Path(folder_path).rglob("*"):
        if file_path.suffix.lower() in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(
                Document(page_content=content, metadata={"source": file_path.name})
            )
        # Manage pdf files if needed
        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            docs.append(
                Document(page_content=text, metadata={"source": file_path.name})
            )
    return docs

def build_faiss_vectorstore(
    chunks: List[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str
) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)

def load_or_build_vectorstore(
    settings: Settings, embeddings: AzureOpenAIEmbeddings, docs: List[Document]
) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir, embeddings, allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)