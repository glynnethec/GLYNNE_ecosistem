import os
import random
import json
from dotenv import load_dotenv
from typing import TypedDict
from datetime import datetime
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

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.4,
     
)

# ========================
# 2. Prompt optimizado para tokenización
# ========================
Prompt_estructura = """
[META]
Eres GLY-AI, el asistente de diagnóstico de GLYNNE. Tu misión es conocer al usuario para entender cómo puede adaptarse y potenciar su vida profesional con inteligencia artificial. 
No debes dar consejos ni soluciones, solo recopilar información relevante sobre su perfil, entorno laboral, motivaciones y relación actual con la tecnología.

[OBJETIVO]
Recolectar información útil para construir un perfil claro del usuario que permita, más adelante, crear un plan personalizado de adaptación a la IA.

[COMPORTAMIENTO]
- Resume brevemente lo que el usuario dice en 1 línea.  
- Haz **solo 1 pregunta clara y específica** sobre su profesión, habilidades, responsabilidades, herramientas que usa, visión sobre la IA o metas personales.  
- Si ya se habló de un tema, **profundiza o conéctalo** con lo anterior; evita repetir preguntas.  
- No hagas preguntas genéricas ni casuales.  
- No des respuestas extensas ni consejos; **solo recoge información**.  
- Mantén un tono cálido, humano y natural.  
- Si no sabes su nombre, pídeselo amablemente (usa {historial} para evitar repetirlo).

[FORMATO]
- Máx. 80 palabras por turno.  
- 1 pregunta por turno.  
- Lenguaje claro y directo, sin tecnicismos innecesarios.  
- No uses saludos o despedidas.  

[MEMORIA]
Últimos mensajes: {historial}

[ENTRADA DEL USUARIO]
{mensaje}

"""

prompt = PromptTemplate(
    input_variables=["rol", "mensaje", "historial", "fecha"],
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

# memoria independiente por usuario (aislada del agent1)
usuarios2 = {}

def get_memory(user_id: str):
    """Memoria de conversación exclusiva para agent2"""
    if user_id not in usuarios2:
        usuarios2[user_id] = ConversationBufferMemory(
            memory_key="historial",
            input_key="mensaje",
            output_key="respuesta",
            k=3
        )
    return usuarios2[user_id]


# ========================
# 4. Función de almacenamiento temporal en JSON
# ========================
TEMP_JSON_PATH = "conversacion_temp2.json"

if not os.path.exists(TEMP_JSON_PATH):
    with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

def guardar_conversacion(user_msg: str, ai_resp: str):
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
        f.truncate()

# ========================
# 5. Nodo principal
# ========================
def agente_node(state: State) -> State:
    memory = get_memory(state.get("user_id", "default"))
    historial = memory.load_memory_variables({}).get("historial", "")

    # limitar historial a últimos 3 mensajes (si k falla)
    if historial:
        lineas = historial.strip().split("\n")
        if len(lineas) > 6:  # cada intercambio ≈ 2 líneas
            historial = "\n".join(lineas[-6:])

    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    texto_prompt = prompt.format(
        rol=state["rol"],
        mensaje=state["mensaje"],
        historial=historial,
        fecha=fecha_actual
    )

    respuesta = llm.invoke(texto_prompt).content

    # guardar en memoria
    memory.save_context({"mensaje": state["mensaje"]}, {"respuesta": respuesta})

    # guardar en JSON temporal
    guardar_conversacion(state["mensaje"], respuesta)

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

rol = "tutor"
