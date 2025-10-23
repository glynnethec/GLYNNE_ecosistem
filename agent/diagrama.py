# agent/ecosistema.py
import os
import random
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("en el .env no hay una api valida")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.4,
)

Prompt_estructura = """
[META]
Analiza la conversación del usuario y construye un ecosistema de gestión de software automatizado con IA.
El ecosistema debe estar representado como un grafo con 15 nodos conectados entre sí.

[NODOS]
Cada nodo representa un módulo de gestión empresarial (ejemplo: Ventas, Finanzas, RRHH, Operaciones, Atención al Cliente, etc.).
Incluye:
- id
- nombre
- descripcion (cómo funciona hoy)
- intervencion_IA (cómo la IA lo transforma)

[RELACIONES]
Cada relación conecta dos nodos (source → target) con una descripción de cómo colaboran mediante IA.

[FORMATO DE SALIDA]
Devuelve ÚNICAMENTE un JSON con esta estructura:
{
  "ecosistema": {
    "nodos": [...],
    "relaciones": [...]
  }
}

[ENTRADA: CONVERSACIÓN AUDITADA]
{conversacion}

respuesta:
"""

prompt = PromptTemplate(
    input_variables=["conversacion"],
    template=Prompt_estructura.strip(),
)

# ========================
# Generador de Ecosistema
# ========================
def generar_ecosistema(conversacion: str) -> dict:
    texto_prompt = prompt.format(conversacion=conversacion)
    respuesta = llm.invoke(texto_prompt).content

    try:
        import json
        return json.loads(respuesta)
    except Exception:
        return {"error": "No se pudo parsear la respuesta a JSON", "raw": respuesta}
