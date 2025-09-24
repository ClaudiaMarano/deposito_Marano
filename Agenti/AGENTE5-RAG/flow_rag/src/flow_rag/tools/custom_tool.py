from typing import Type
import os
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from flow_rag.tools.utility import (
    SETTINGS,
    get_embeddings,
    load_documents_from_folder,
    load_or_build_vectorstore,
    make_retriever,
    get_contexts_for_question
)


class MyToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    query: str = Field(..., description="This is the input query used to retrieve relevant information.")
    #k: int = Field(..., description="Number of the most relevant chunks to retrieve.")

class MyCustomTool(BaseTool):
    name: str = "get_contexts_tool"
    description: str = "Fetch relevant contexts for a given question."
    args_schema: Type[BaseModel] = MyToolInput

    def _run(self, query: str) -> str:
        try:
            # Get the absolute path to the data folder
            current_dir = Path(__file__).parent.parent
            data_path = current_dir / "data"
            
            # Initialize the RAG system
            embeddings = get_embeddings(SETTINGS)
            docs = load_documents_from_folder(str(data_path))
            vector_store = load_or_build_vectorstore(SETTINGS, embeddings, docs)
            retriever = make_retriever(vector_store, SETTINGS)
            
            # Get contexts and return
            contexts = get_contexts_for_question(retriever, query, SETTINGS.k)
            return "\n\n".join(contexts)
        except Exception as e:
            return f"Error retrieving contexts: {str(e)}"


