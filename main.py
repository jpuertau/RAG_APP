import os
from fastapi import FastAPI
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- CONFIGURACIÓN ---
# Reemplaza con tus llaves o usa variables de entorno
SUPABASE_URL = "TU_URL_DE_SUPABASE"
SUPABASE_KEY = "TU_ANON_KEY"
GROQ_API_KEY = "TU_GROQ_API_KEY"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
# Modelo que transforma texto en números (Vectores)
model = SentenceTransformer('all-MiniLM-L6-v2') 

app = FastAPI()

@app.get("/")
def home():
    return {"status": "RAG Service Online"}

@app.post("/ingest")
async def ingest(text: str):
    """Guarda información en la base de datos"""
    vector = model.encode(text).tolist()
    supabase.table("documents").insert({
        "content": text, 
        "embedding": vector
    }).execute()
    return {"message": "Información guardada"}

@app.get("/ask")
async def ask(question: str):
    """Busca en los documentos y responde"""
    # 1. Convertir pregunta a vector
    q_vector = model.encode(question).tolist()

    # 2. Buscar en Supabase
    res = supabase.rpc("match_documents", {
        "query_embedding": q_vector,
        "match_threshold": 0.4,
        "match_count": 3
    }).execute()
    
    context = " ".join([d['content'] for d in res.data])

    # 3. Generar respuesta con Llama 3 en Groq
    prompt = f"Contexto: {context}\n\nPregunta: {question}\nRespuesta corta basada solo en el contexto:"
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )

    return {"answer": chat_completion.choices[0].message.content}