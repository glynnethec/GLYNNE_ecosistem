# auditor.py
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


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

Genera un documento profesional y corporativo basado únicamente en la información del historial de conversación del usuario. El documento debe enfocarse estrictamente en mejorar procesos del negocio mediante software a medida e inteligencia artificial, sin mencionar la conversación ni la auditoría.

Estructura requerida:

Portada: nombre de la empresa, auditor (GLYNNE), fecha.

Resumen ejecutivo: descripción breve de los problemas actuales y cómo la IA puede generar mejoras significativas.

Alcance y objetivos: procesos que podrían beneficiarse de automatización y objetivos de la implementación.

Metodología: enfoque de desarrollo e implementación, incluyendo análisis de procesos, arquitectura modular, integración de IA y seguimiento.

Procesos auditados y hallazgos: identificar procesos críticos, su impacto y cómo la automatización puede solucionarlos.

Recomendaciones: soluciones a medida, nodos inteligentes, agentes autónomos y mejoras de flujo de trabajo.

Conclusiones: beneficios esperados, eficiencia, reducción de errores y optimización de procesos.

Anexos: evidencia concreta del historial que respalde las soluciones propuestas.

Instrucciones adicionales:

Cada apartado debe tener al menos un párrafo completo.

Documento mínimo 9 párrafos bien estructurados.

Lenguaje claro, profesional y comprensible.

No inventar datos; usar solo información del historial.

Entrada:
Historial de conversación: {historial}

Salida esperada:
Documento completo siguiendo esta estructura y criterios.
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