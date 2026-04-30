# main.py
import os
from fastapi import FastAPI
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. Cargar configuración
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# 2. Inicializar clientes (esto es rápido)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# 3. Cargar el modelo (esto puede tardar, lo ponemos fuera de las rutas)
# Usamos un modelo aún más pequeño para asegurar que quepa en la RAM de Render
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"} # Esto ayuda a Render a saber que la app vive