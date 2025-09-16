import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

st.title("Chatbot Embedding con Azure OpenAI")

# Input dell'utente
user_input = st.text_input("Inserisci una frase per ottenere l'embedding:")

if st.button("Genera embedding") and user_input:
    try:
        # Richiesta di embedding
        response = client.embeddings.create(
            model="text-embedding-ada-002",  # nome del deployment creato su Azure
            input=user_input
        )
        
        embedding = response.data[0].embedding
        
        # Mostra il risultato
        st.success("Embedding generato con successo!")
        st.write("Lunghezza embedding:", len(embedding))
        st.write("Primi 10 valori:", embedding[:10])
        
        # Opzione per mostrare il vettore completo
        if st.checkbox("Mostra vettore completo"):
            st.write("Vettore completo:", embedding)
            
    except Exception as e:
        st.error(f"Errore durante la generazione dell'embedding: {e}")