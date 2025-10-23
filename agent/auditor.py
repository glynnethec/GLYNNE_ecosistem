# auditor.py
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ========================
# 1. Cargar entorno y API
# ========================
load_dotenv()
api_key = os.getenv('GROQ_API_KEY2')
hf_api_key = os.getenv('HUGGINGFACE_API_KEY2')

if not api_key:
    raise ValueError('No hay una API key válida de Groq en el .env')

if not hf_api_key:
    raise ValueError('No hay una API key válida de Hugging Face en el .env')

# ========================
# 2. Inicializar LLM principal (Groq)
# ========================
llm = ChatGroq(
    model="Llama-3.1-8B-Instant",
    api_key=api_key,
    temperature=0.7,

)

# ========================
# 2b. LLM de fallback Hugging Face vía APIs
# ========================
def llm_huggingface_fallback(prompt_text: str) -> str:
    """
    Fallback a Hugging Face usando API Key
    """
    try:
        from transformers import pipeline

        generator = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            device=-1,
            use_auth_token=hf_api_key
        )

        output = generator(
            prompt_text,
            max_length=500,
            do_sample=True,
            top_p=0.95
        )
        return output[0]["generated_text"]

    except Exception as e:
        print("❌ Error fallback Hugging Face:", e)
        return "Lo siento, no pude generar la auditoría."

# ========================
# 3. Prompt de auditoría
# ========================
Prompt_estructura = """
[META]
Fecha del reporte: {fecha}

Tu meta es analizar el negocio del usuario usando únicamente la información proporcionada en la conversación histórica. Genera un documento profesional y corporativo, **estrictamente enfocado en cómo mejorar los procesos del negocio mediante software personalizado e inteligencia artificial para automatización**. No incluyas información sobre la conversación en sí, ni sobre cómo se hizo la auditoría. Solo centra el documento en soluciones.

El documento debe tener los siguientes apartados:

1. **Portada**: nombre de la empresa del usuario, auditor (GLYNNE), fecha.
2. **Resumen ejecutivo**: breve descripción de los problemas actuales del negocio y cómo un software impulsado por IA puede generar mejoras significativas.
3. **Alcance y objetivos**: delimitar los procesos que podrían beneficiarse de la automatización y la optimización con IA, y los objetivos de la implementación.
4. **Metodología**: enfoque propuesto para el desarrollo e implementación del software de automatización, incluyendo análisis de procesos, diseño de arquitectura modular, integración de IA, y seguimiento.
5. **Procesos auditados y hallazgos**: identificar los procesos críticos que presentan problemas o ineficiencias, el impacto de esos problemas y cómo la automatización mediante IA puede resolverlos.
6. **Recomendaciones**: propuestas concretas de soluciones a medida, integraciones de IA, nodos inteligentes, agentes autónomos y mejoras de flujo de trabajo.
7. **Conclusiones**: beneficios esperados del software a medida, eficiencia, reducción de errores, y optimización de procesos.
8. **Anexos**: evidencia relevante de los puntos mencionados (solo fragmentos de información del usuario que respalden las soluciones propuestas).
9.minimo 9 parrafos bien estructurados 
Cada apartado debe tener al menos un párrafo completo, explicando claramente la situación, impacto y posibles mejoras. **No inventes datos**. Usa solo la información proporcionada por el usuario en el historial de conversación.

[ENTRADA DEL USUARIO]
Historial de conversación: {historial}

Respuesta:
"""


prompt_template = PromptTemplate(
    input_variables=["historial", "fecha"],
    template=Prompt_estructura.strip()
)

# ========================
# 4. Función para generar auditoría
# ========================
def generar_auditoria():
    json_path = "conversacion_temp.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encontró el archivo: {json_path}")

    # Leer conversación
    with open(json_path, "r", encoding="utf-8") as f:
        conversacion = json.load(f)

    # Formatear conversación
    historial_texto = ""
    for intercambio in conversacion:
        historial_texto += f"Usuario: {intercambio.get('user', '')}\n"
        historial_texto += f"GLY-AI: {intercambio.get('ai', '')}\n"

    # Obtener fecha actual
    fecha_actual = datetime.now().strftime("%d/%m/%Y")

    # Crear prompt con fecha
    prompt_text = prompt_template.format(historial=historial_texto, fecha=fecha_actual)

    # Llamar LLM principal con fallback
    try:
        respuesta = llm.invoke(prompt_text)
        texto_final = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
    except Exception as e:
        print("❌ Error en Groq LLM:", e)
        texto_final = llm_huggingface_fallback(prompt_text)

    # === Limpiar el archivo JSON después de usarlo ===
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("✅ Archivo de conversación limpiado después de generar la auditoría.")
    except Exception as e:
        print("❌ Error al limpiar el archivo JSON:", e)

    return texto_final

# ========================
# 5. CLI opcional para pruebas
# ========================
if __name__ == "__main__":
    print("LLM Auditoría iniciado")
    try:
        resultado = generar_auditoria()
        print("\n===== AUDITORÍA =====\n")
        print(resultado)
        print("\n=====================\n")
    except Exception as e:
        print("❌ Error general:", e)