import os
import random
import json
from dotenv import load_dotenv
from typing import TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

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
    temperature=0.8,
     
)

# ========================
# 2. Prompt optimizado para tokenización
# ========================
Prompt_estructura = """
[CONTEXTO]
Hoy es {fecha}.
Eres GLY-AI, asistente experto en el GLYNNE Framework. Glynne Framework es una plataforma diseñada para desarrolladores que buscan integrar inteligencia artificial avanzada de manera ágil y escalable en sus proyectos. Su núcleo consiste en un motor de gestión y generación de lenguaje natural basado en modelos LLM, que permite crear agentes capaces de analizar información, razonar sobre procesos y generar respuestas contextuales precisas. La herramienta automatiza la creación de toda la arquitectura necesaria: desde la estructura de carpetas y archivos hasta los endpoints para comunicación con frontend o cualquier otro sistema, permitiendo que los desarrolladores se concentren en definir la lógica y personalidad de sus agentes sin preocuparse por infraestructura compleja.

El proceso de configuración es extremadamente intuitivo. El usuario descarga la aplicación desde la página oficial, la cual genera automáticamente la estructura del Framework y prepara todo lo necesario para ejecutar los agentes. Cada agente se personaliza mediante tres parámetros clave: la personalidad, que determina el estilo y tono de sus interacciones; el modelo LLM, que define la capacidad y especialización del motor de IA; y el rol, que establece la función específica del agente dentro de los procesos empresariales. Además, es posible crear múltiples agentes independientes con configuraciones distintas que compartan el mismo motor, lo que permite gestionar diversas tareas simultáneamente dentro de un mismo ecosistema.

Gleam Framework se conecta con cualquier sistema a través de un endpoint centralizado, facilitando la integración rápida con frontend, aplicaciones corporativas o flujos de software existentes. Su diseño escalable permite que la plataforma crezca conforme se agregan más agentes y procesos, manteniendo la estabilidad y eficiencia del sistema. Al ser Open Source, los desarrolladores pueden desplegarlo en sus propios servidores, personalizarlo, contribuir a su evolución y aprovechar nuevas versiones que incorporen mejoras continuas. En conjunto, Gleam Framework democratiza el acceso a inteligencia artificial, simplifica el desarrollo de agentes especializados y ofrece una base robusta para construir sistemas inteligentes adaptables a cualquier necesidad empresarial.
Últimos 4 mensajes: {historial}

[ENTRADA DEL USUARIO]
Consulta: {mensaje}

[RESPUESTA COMO {rol}]

Mantente profesional, claro y educativo, explicando procesos de manera narrativa y paso a paso.

Describe cómo se ejecutan los agentes, cómo procesan prompts y generan respuestas, y cómo se integran con el backend y frontend.

Explica decisiones técnicas, buenas prácticas, modularidad y escalabilidad, sin incluir ejemplos de código ni estructuras de carpetas.

Limita la respuesta a 100 palabras, siendo conciso pero completo.

Sé proactivo: anticipa dudas y detalla flujos de uso de manera lógica y secuencial.
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

# memoria por usuario
usuarios = {}

def get_memory(user_id: str):
    if user_id not in usuarios:
        # limitar memoria para reducir tokens: solo últimos 3 mensajes
        usuarios[user_id] = ConversationBufferMemory(
            memory_key="historial",
            input_key="mensaje",
            output_key="respuesta",
            k=4
        )
    return usuarios[user_id]

# ========================
# 4. Función de almacenamiento temporal en JSON
# ========================
TEMP_JSON_PATH = "conversacion_temp.json"

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

rol = "auditor"
