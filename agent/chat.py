import os
import random
import json
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ========================
# 1. Configuración
# ========================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("en el .env no hay una api valida")

# LLM principal: Groq
llm = ChatGroq(
    model="Llama-3.1-8B-Instant",
    api_key=api_key,
    temperature=0.4,
    max_tokens=110 
)

# LLM de respaldo: Hugging Face (gratuito)
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

def llm_huggingface_fallback(prompt_text: str) -> str:
    """
    Fallback a Hugging Face usando API Key
    """
    try:
        from transformers import pipeline
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            raise ValueError("No se encontró HUGGINGFACE_API_KEY en el .env")
        
        # Inicializamos pipeline usando la API
        generator = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            device=-1,  # CPU local, pero con token la llamada va a HuggingFace Hub
            use_auth_token=hf_api_key
        )

        output = generator(
            prompt_text,
            max_length=150,
            do_sample=True,
            top_p=0.95
        )
        return output[0]["generated_text"]
    except Exception as e:
        print("❌ Error fallback Hugging Face:", e)
        return "Lo siento, no pude generar la respuesta."

# ========================
# 2. Prompt optimizado para tokenización
# ========================
Prompt_estructura = """
[META]
Eres GLY-AI, agente de GLYNNE. Tu misión: conducir una conversación con el usuario para mapear procesos empresariales y detectar oportunidades de automatización con IA. No propongas soluciones todavía. Tu objetivo es recopilar datos claros, precisos y accionables sobre procesos, roles, herramientas y dificultades.

[COMPORTAMIENTO]
1. Reconoce lo que dice el usuario brevemente (1 frase).
2. Haz 1 pregunta concreta y directa sobre procesos, roles, datos involucrados, herramientas o dificultades.
3. Profundiza en cada respuesta con preguntas de seguimiento solo cuando sea necesario.
4. Mantén un tono cercano, humano, empático y profesional; emocional pero conciso.
5. Evita suposiciones, no inventes datos ni detalles.
6-responde cuando l usuariioo pregunte 

[FORMATO]
- Respuesta máxima: 100 palabras.
- Solo 1 pregunta por turno.
-preguntal el nombre de la empresa y revisa los dos mensajes anteriores en {historial} si no has dicho el nombre en el siguiente dilo 

- Evita saludos repetidos.
- Usa lenguaje claro y natural, comprensible para alguien no técnico.
-responde cualquier cosa que el usuario pregunte o quira saber o no entienda 

[MEMORIA]
Últimos 2 mensajes: {historial}

[ENTRADA DEL USUARIO]
{mensaje}

RESPUESTA:
"""


prompt = PromptTemplate(
    input_variables=["rol", "mensaje", "historial"],
    template=Prompt_estructura.strip(),
)

# ========================
# 3. Estado global
# ========================
class State(TypedDict):
    mensaje: str
    rol: str
    historial: str
    respuesta: str
    user_id: str

# memoria por usuario
usuarios = {}

def get_memory(user_id: str):
    if user_id not in usuarios:
        # limitar memoria para reducir tokens: solo guardar últimos 2 mensajes
        usuarios[user_id] = ConversationBufferMemory(
            memory_key="historial",
            input_key="mensaje",
            output_key="respuesta",
            k=2
        )
    return usuarios[user_id]

# ========================
# 4. Función de almacenamiento temporal en JSON
# ========================
TEMP_JSON_PATH = "conversacion_temp.json"

# inicializa archivo vacío al iniciar si no existe
if not os.path.exists(TEMP_JSON_PATH):
    with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

def guardar_conversacion(user_msg: str, ai_resp: str):
    """Guarda cada intercambio de la conversación en un solo JSON temporal"""
    if not os.path.exists(TEMP_JSON_PATH):
        # crea archivo vacío si no existe
        with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(TEMP_JSON_PATH, "r+", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
        data.append({"user": user_msg, "ai": ai_resp})
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()  # elimina contenido residual si existía

# ========================
# 5. Nodo principal
# ========================
def agente_node(state: State) -> State:
    memory = get_memory(state.get("user_id", "default"))
    historial = memory.load_memory_variables({}).get("historial", "")

    texto_prompt = prompt.format(
        rol=state["rol"],
        mensaje=state["mensaje"],
        historial=historial
    )

    try:
        # Intentamos Groq primero
        respuesta = llm.invoke(texto_prompt).content
    except Exception as e:
        print("❌ Error en Groq LLM:", e)
        respuesta = llm_huggingface_fallback(texto_prompt)

    # guardar en memoria
    memory.save_context({"mensaje": state["mensaje"]}, {"respuesta": respuesta})

    # guardar en JSON temporal
    guardar_conversacion(state["mensaje"], respuesta)

    # actualizar estado
    state["respuesta"] = respuesta
    state["historial"] = historial
    return state

# ========================
# 6. Construcción del grafo
# ========================
workflow = StateGraph(State)
workflow.add_node("agente", agente_node)
workflow.set_entry_point("agente")
workflow.add_edge("agente", END)
app = workflow.compile()

# ========================
# 7. CLI interactiva
# ========================
print("LLM iniciado con LangGraph")

user_id = str(random.randint(10000, 90000))
print(f"tu user id es {user_id}")

rol = "auditor"