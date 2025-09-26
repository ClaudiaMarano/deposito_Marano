from __future__ import annotations
 
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
import bs4
import re
 
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader, UnstructuredImageLoader, WebBaseLoader
from duckduckgo_search import DDGS
 
# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
 
# Chat model init (provider-agnostic, qui puntiamo a LM Studio via OpenAI-compatible)
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
 
# --- RAGAS ---
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # "precision@k" sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_relevancy,    # pertinenza della risposta vs domanda
    answer_correctness,  # usa questa solo se hai ground_truth
)
 
 
# =========================
# Configurazione
# =========================
 
load_dotenv()
 
@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "faiss_index_anberlino"
    # Text splitting
    chunk_size: int = 200
    chunk_overlap: int = 100
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 4                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3         # 0 = diversificazione massima, 1 = pertinenza massima
    # Embedding
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LM Studio (OpenAI-compatible)
    lmstudio_model_env: str = "gpt-4o"  # nome del modello in LM Studio, via env var
    azure_embedding_model: str = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-ada-002")
    api_version: str = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    azure_endpoint: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key: str | None = os.getenv("AZURE_OPENAI_API_KEY")
   
 
 
 
SETTINGS = Settings()
 
 
# =========================
# Componenti di base
# =========================
 
def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Ritorna i testi dei top-k documenti (chunk) usati come contesto."""
    docs = docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]
 
def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Esegue la pipeline RAG per ogni domanda e costruisce il dataset per Ragas.
    Ogni riga contiene: question, contexts, answer, (opzionale) ground_truth.
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        answer = chain.invoke(q)
 
        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]
 
        dataset.append(row)
    return dataset
 
def get_azure_embedding_model(settings: Settings = SETTINGS) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-ada-002"),
        model=os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION", "2023-05-15"),
        # chunk_size=settings.chunk_size,
        validate_base_url=True,
    )
 
 
def get_llm_from_lmstudio():
    """
    Inizializza un ChatModel puntando a LM Studio (OpenAI-compatible).
    Richiede:
      - OPENAI_BASE_URL (es. http://localhost:1234/v1)
      - OPENAI_API_KEY (placeholder qualsiasi, es. "not-needed")
      - LMSTUDIO_MODEL (nome del modello caricato in LM Studio)
    """
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    model_name = os.getenv("AZURE_MODEL")
 
    if not base_url or not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_API_KEY devono essere impostate per Azure OpenAI"
        )
    if not model_name:
        raise RuntimeError(
            f"Imposta la variabile AZURE_MODEL con il nome del modello caricato in Azure"
        )
 
    # model_provider="openai" perché l'endpoint è OpenAI-compatible
    return init_chat_model(model_name, api_key=api_key, api_version=api_version, model_provider="azure_openai")
 
 
def simulate_corpus() -> List[Document]:
    """
    Crea un piccolo corpus di documenti in inglese con metadati e 'source' per citazioni.
    """
    docs = [
        Document(
            page_content=(
                "LangChain is a framework that helps developers build applications "
                "powered by Large Language Models (LLMs). It provides chains, agents, "
                "prompt templates, memory, and integrations with vector stores."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md"}
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search and clustering of dense vectors. "
                "It supports exact and approximate nearest neighbor search and scales to millions of vectors."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md"}
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce sentence embeddings suitable "
                "for semantic search, clustering, and information retrieval. The embedding size is 384."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md"}
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store) and "
                "retrieval+generation. Retrieval selects the most relevant chunks, and the LLM produces "
                "an answer grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md"}
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) balances relevance and diversity during retrieval. "
                "It helps avoid redundant chunks and improves coverage of different aspects."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md"}
        ),
    ]
    return docs
 
def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Carica documenti da file PDF, CSV e immagini, restituendo una lista di Document.
    """
    print(f"🔍 LOAD_DOCUMENTS: Caricamento di {len(file_paths)} file(s)")
    documents = []
    for file_path in file_paths:
        print(f"📄 Caricamento file: {file_path}")
        ext = file_path.split('.')[-1].lower()
        if ext == 'pdf':
            loader = PyMuPDFLoader(file_path)
        elif ext == 'csv':
            loader = CSVLoader(file_path)
        elif ext in ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff']:
            loader = UnstructuredImageLoader(file_path)
        else:
            print(f"❌ Tipo file non supportato: {file_path}")
            continue
        docs = loader.load()
        print(f"   ✅ Caricati {len(docs)} documento/i da {file_path}")
        for i, doc in enumerate(docs):
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"      Doc {i+1}: {len(doc.page_content)} caratteri - '{content_preview}...'")
        documents.extend(docs)
   
    print(f"📚 LOAD_DOCUMENTS: Totale {len(documents)} documenti caricati")
    return documents
 
 
def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)
 
 
def build_faiss_vectorstore(chunks: List[Document], embeddings: get_azure_embedding_model, persist_dir: str) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
 
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs
 
 
def load_or_build_vectorstore(settings: Settings, embeddings: get_azure_embedding_model, docs: List[Document]) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"
 
# if index_file.exists() and meta_file.exists():
#     # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
#     return FAISS.load_local(
#         settings.persist_dir,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
 
    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)
 
 
def make_retriever(vector_store: FAISS, settings: Settings):
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )
 
 
def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)
 
 
def build_rag_chain(llm, retriever):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi in italiano. "
        "Usa esclusivamente il CONTENUTO fornito nel contesto. "
        "Se l'informazione non è presente, dichiara che non è disponibile. "
        "Includi citazioni tra parentesi quadre nel formato [source:...]. "
        "Sii conciso, accurato e tecnicamente corretto."
    )
 
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi solo con informazioni contenute nel contesto. Immedesimati in un mondo fittizio dove le informazioni vere sono quelle del contesto fornito. \n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.' \n"
         "4) Se la risposta è ambigua o incompleta, chiedi ulteriori chiarimenti all'utente. \n"
         "5) Se la domanda riguarda informazioni contrarie al contesto, scrivi la risposta corretta secondo il contesto.\n")
    ])
 
    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
 
 
def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)
 
def scan_docs_folder(docs_dir: str = "docs") -> List[str]:
    """
    Scansiona la cartella docs e restituisce tutti i file supportati.
    """
    supported_extensions = {'.pdf', '.csv', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    file_paths = []
   
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"❌ Cartella {docs_dir} non trovata")
        return []
   
    # Scansione ricorsiva
    for file_path in docs_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            file_paths.append(str(file_path))
   
    print(f"📂 Trovati {len(file_paths)} file nella cartella {docs_dir}")
    return file_paths
 
def keywords_generation(query: str):
    llm = get_llm_from_lmstudio()
   
    prompt = f"""You are a helpful assistant. Generate keywords separated with commas for web search based on the user's query.
   
    Query: {query}
   
    Keywords:"""
   
    response = llm.invoke(prompt)
    keywords = response.content.strip().split(", ")  
    print(keywords)
    return keywords
 
def ddgs_results(query: str, max_results: int = 5):
    """
    Ricerca web con DDGS diretto, bypassa SSL e gestisce errori.
    """
    print(f"🔍 DDGS: Ricerca per '{query}' (max {max_results} risultati)")
   
    try:
        # DDGS con SSL bypass
        with DDGS(
            verify=False,  # ← BYPASSA SSL
            timeout=20,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; DDGSBot/1.0)'}
        ) as ddgs:
           
            results = list(ddgs.text(
                keywords=query,
                max_results=max_results,
                region='it-it',
                safesearch='moderate',
                timelimit='y'  # Risultati dell'ultimo anno
            ))
       
        print(f"   ✅ {len(results)} risultati trovati")
       
        # Formatta per compatibilità con il resto del codice
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(result.get('href', ''))
            print(f"   {i}. {result.get('title', '')[:50]}...")
       
        return formatted
       
    except Exception as e:
        print(f"❌ Errore DDGS: {e}")
        return []
 
def clean_web_content(text: str) -> str:
    """
    Pulisce il contenuto web da elementi indesiderati.
    """
    if not text:
        return ""
   
    # Rimuovi caratteri di controllo e spazi multipli
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
   
    # Rimuovi pattern comuni di navigazione e UI
    patterns_to_remove = [
        r'Cookie Policy|Privacy Policy|Note Legali|Termini e Condizioni',
        r'Accetta tutti i cookie|Gestisci cookie|Rifiuta cookie',
        r'Iscriviti alla newsletter|Seguici su|Condividi su',
        r'Copyright.*?\d{4}|All rights reserved|Tutti i diritti riservati',
        r'Menu|Navbar|Header|Footer|Sidebar',
        r'Caricamento in corso|Loading|Attendere prego',
        r'Clicca qui|Click here|Leggi tutto|Read more',
        r'Ti potrebbe interessare|Articoli correlati|Notizie correlate',
        r'I più visti|Più letti|Trending|Popular',
        r'Pubblicità|Advertisement|Sponsor|Promo',
        r'PODCAST|RUBRICHE|SONDAGGI|LE ULTIME EDIZIONI',
        r'Ascolta i Podcast.*?|Vedi tutti.*?|Scopri di più.*?'
    ]
   
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
   
    # Rimuovi URL e email
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
   
    # Rimuovi numeri isolati (spesso date, ore, contatori)
    text = re.sub(r'\b\d{1,2}[:.]\d{2}\b', '', text)  # Orari
    text = re.sub(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b', '', text)  # Date
   
    # Rimuovi caratteri speciali ripetuti
    text = re.sub(r'[^\w\s.,!?;:()\-"\'àèéìíîòóùúçñü]+', ' ', text, flags=re.UNICODE)
   
    # Rimuovi linee molto corte (probabilmente navigazione)
    lines = text.split('.')
    meaningful_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 20 and not re.match(r'^[A-Z\s]+$', line):  # Non solo maiuscole
            meaningful_lines.append(line)
   
    text = '. '.join(meaningful_lines)
   
    # Pulizia finale
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
   
    return text
 
def web_search_and_format(path: str):
    """
    Esegue una ricerca web, carica il contenuto e lo pulisce per un migliore processing.
    Se WebBaseLoader fallisce (es. problemi SSL), facciamo fallback a requests (verify=False)
    e parsifichiamo l'HTML con BeautifulSoup per estrarre il testo.
    """
    print(f"🌐 Caricamento contenuto da: {path}")

    try:
        # Proviamo prima con WebBaseLoader e vari selettori
        selectors = [
            ("article", ["article"]),
            ("main-content", ["main", ".main", "#main"]),
            ("content-areas", [".content", ".post-content", ".article-content", ".entry-content"]),
            ("text-body", [".text", ".body", ".story-body", ".article-body"]),
        ]

        valid_docs: list[Document] = []

        for name, elems in selectors:
            if valid_docs:
                break
            try:
                loader = WebBaseLoader(web_paths=(path,), bs_kwargs=dict(parse_only=bs4.SoupStrainer(elems)))
                docs = loader.load()
                print(f"   🔍 Tentativo con selettori {name}: {len(docs)} documenti")
                for doc in docs:
                    cleaned = clean_web_content(doc.page_content)
                    if len(cleaned) > 100:
                        doc.page_content = cleaned
                        valid_docs.append(doc)
                        print(f"   ✅ Contenuto valido trovato: {len(cleaned)} caratteri puliti")
                        break
            except Exception as e:
                print(f"   ⚠️ Errore con selettori {name}: {e}")
                continue

        # Se non abbiamo ancora contenuto, fallback via requests (disabilitiamo la verifica SSL)
        if not valid_docs:
            print("   🔄 Nessun contenuto valido trovato con WebBaseLoader, provo fallback diretto con requests (verify=False)...")
            try:
                import requests
                headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
                resp = requests.get(path, headers=headers, timeout=20, verify=False)
                html = resp.text
                soup = bs4.BeautifulSoup(html, "html.parser")

                # Cerca nodi preferiti
                candidate_text = None
                for sel in ["article", "main", "#main", ".main", ".content", ".post-content", ".entry-content"]:
                    node = soup.select_one(sel)
                    if node and node.get_text(strip=True):
                        candidate_text = node.get_text(separator=' ', strip=True)
                        break

                # Fallback al body o all'intero documento
                if not candidate_text:
                    if soup.body and soup.body.get_text(strip=True):
                        candidate_text = soup.body.get_text(separator=' ', strip=True)
                    else:
                        candidate_text = soup.get_text(separator=' ', strip=True)

                cleaned = clean_web_content(candidate_text or "")
                if len(cleaned) > 150:
                    valid_docs.append(Document(page_content=cleaned, metadata={"source": path}))
                    print(f"   ✅ Contenuto recuperato via requests: {len(cleaned)} caratteri")
                else:
                    print("   ⚠️ Contenuto recuperato ma troppo corto dopo pulizia")
            except Exception as e:
                print(f"   ❌ Fallback requests fallito: {e}")

        # Se ancora vuoto, ritorna messaggio di fallback
        if not valid_docs:
            print("   ❌ Impossibile estrarre contenuto significativo")
            return [Document(
                page_content=f"Contenuto non disponibile per {path}. Il sito web potrebbe non essere accessibile o non contenere testo leggibile.",
                metadata={"source": path, "error": "no_meaningful_content"}
            )]

        # Mostra preview
        for i, doc in enumerate(valid_docs):
            preview = doc.page_content[:200].replace('\n', ' ')
            print(f"   📄 Preview contenuto {i+1}: '{preview}...'")
        return valid_docs

    except Exception as e:
        print(f"❌ Errore generale nel caricamento di {path}: {e}")
        return [Document(page_content=f"Errore nel caricamento di {path}: {str(e)}", metadata={"source": path, "error": str(e)})]
 
def ragas_evaluation(question: str, chain, llm, embeddings, retriever, settings: Settings = SETTINGS):
        questions = [question]
        dataset = build_ragas_dataset(
        questions=questions,
        retriever=retriever,
        chain=chain,
        k=settings.k
    )
 
        evaluation_dataset = EvaluationDataset.from_list(dataset)
 
        # 7) Scegli le metriche
        metrics = [
                #     context_precision,
                #    context_recall,
                   faithfulness,
                   answer_relevancy
                   ]
        # Aggiungi correctness solo se tutte le righe hanno ground_truth
        if all("ground_truth" in row for row in dataset):
            metrics.append(answer_correctness)
 
        # 8) Esegui la valutazione con il TUO LLM e le TUE embeddings
        ragas_result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=llm,                 # passa l'istanza LangChain del tuo LLM (LM Studio)
            embeddings=embeddings,  # o riusa 'embeddings' creato sopra
        )
 
        df = ragas_result.to_pandas()
        cols = ["faithfulness", "answer_relevancy"]
        return df[cols].round().to_string(index=False)
 
# =========================
# Esecuzione dimostrativa
# =========================
 
def main():
    settings = SETTINGS
 
 
    print("\n=== METRICHE AGGREGATE ===")
 
 
    # 1) Componenti
    embeddings = get_azure_embedding_model()
    llm = get_llm_from_lmstudio()
 
    # 2) Dati simulati e indicizzazione (load or build)
    req = input("Vuoi eseguire una ricerca web per arricchire il contesto? (s/n): ").strip().lower()
    if req == 'n':
        # docs = simulate_corpus()
        file_paths = scan_docs_folder("docs")
        docs = load_documents(file_paths)
        vector_store = load_or_build_vectorstore(settings, embeddings, docs)
    else:
        query = input("Inserisci il termine di ricerca web: ").strip()
        print(f"🔍 Eseguo ricerca web su: {query}")
        keywords = keywords_generation(query)
        print(f"🔑 Keywords generate: {keywords}")
        str_new = ' '.join(keywords)
        ddgs_request = ddgs_results(str_new)
        print(ddgs_request)
        all_docs = []
        for url in ddgs_request:
            docs = web_search_and_format(url)
            all_docs.extend(docs)
        print(f"📚 Caricati {len(all_docs)} documenti dalla ricerca web")
        vector_store = load_or_build_vectorstore(settings, embeddings, all_docs)
 
    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)
 
    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)
 
 
 
 
 
    while True:
        question = input("Inserisci la tua domanda (o 'exit'/'quit'/'q' per uscire): ")
       
        # Controlla se l'utente vuole uscire
        if question.lower().strip() in ["exit", "quit", "q", "esci"]:
            print("👋 Arrivederci!")
            break
           
        print("=" * 80)
        print("Q:", question)
        print("-" * 80)
        ans = rag_answer(question, chain)
        print(ans)
        print()
   
        rag_eval = ragas_evaluation(question, chain, llm, embeddings, retriever, settings)
        print("\n METRICHE OTTENUTE:\n", rag_eval)
 
 
if __name__ == "__main__":
    main()