import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

import pandas as pd

# ========================
# 1. Configuración
# ========================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("No hay API key en el .env")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.3,
)

# ========================
# 2. Función de análisis CSV (tu código original)
# ========================
def analizar_csv(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        
        # Limpieza básica
        df = df.dropna(axis=1, how='all')
        
        # Detectar tipos de columnas
        numericas = df.select_dtypes(include=['number'])
        categoricas = df.select_dtypes(include=['object'])
        fechas = df.select_dtypes(include=['datetime64', 'datetime'])
        
        reporte = []
        
        # Resumen de columnas numéricas
        if not numericas.empty:
            desc_num = numericas.describe().T
            reporte.append("📊 **RESUMEN NUMÉRICO:**")
            for col, row in desc_num.iterrows():
                reporte.append(f"- {col}: media={row['mean']:.2f}, min={row['min']}, max={row['max']}, nulos={df[col].isna().sum()}")
        else:
            reporte.append("⚠️ No hay columnas numéricas")
        
        # Resumen de columnas categóricas
        if not categoricas.empty:
            reporte.append("\n🔤 **RESUMEN CATEGÓRICO:**")
            for col in categoricas.columns:
                top_val = df[col].value_counts().head(3)
                reporte.append(f"- {col}: {df[col].nunique()} valores únicos. Top 3: {', '.join(top_val.index.astype(str))}")
        else:
            reporte.append("\n⚠️ No hay columnas categóricas")
        
        # Resumen de fechas
        if not fechas.empty:
            reporte.append("\n📅 **RESUMEN FECHAS:**")
            for col in fechas.columns:
                reporte.append(f"- {col}: rango {df[col].min()} → {df[col].max()}, nulos={df[col].isna().sum()}")
        
        # Valores faltantes
        nulls = df.isna().sum()
        cols_with_nulls = nulls[nulls > 0]
        if not cols_with_nulls.empty:
            reporte.append("\n❗ **VALORES FALTANTES:**")
            for col, val in cols_with_nulls.items():
                reporte.append(f"- {col}: {val} nulos ({val/len(df)*100:.1f}%)")
        
        # Información general
        reporte.append(f"\n📈 **ESTADÍSTICAS GENERALES:**")
        reporte.append(f"- Filas: {len(df)}")
        reporte.append(f"- Columnas: {len(df.columns)}")
        reporte.append(f"- Memoria usada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return "\n".join(reporte)
        
    except Exception as e:
        return f"❌ Error procesando el CSV: {str(e)}"

# ========================
# 3. Prompt para análisis avanzado
# ========================
prompt_analisis = """
Eres un analista de datos senior. Analiza el siguiente reporte técnico de un dataset y genera un informe ejecutivo completo.

**DATOS TÉCNICOS DEL DATASET:**
{datos_csv}

**INSTRUCCIONES PARA EL ANÁLISIS:**
1. **PATRONES IDENTIFICADOS**: ¿Qué patrones, tendencias o correlaciones observas?
2. **PROBLEMAS DETECTADOS**: ¿Qué datos están incompletos, inconsistentes o requieren atención?
3. **OPORTUNIDADES**: ¿Qué insights valiosos se pueden extraer?
4. **RECOMENDACIONES**: Acciones concretas para mejorar la calidad de datos y el análisis

**FORMATO DE RESPUESTA:**
- Sé específico y basado únicamente en los datos proporcionados
- No inventes información que no esté en el reporte
- Usa un lenguaje claro y profesional
- Incluye porcentajes y métricas cuando sea posible

**INFORME EJECUTIVO:**
"""

prompt = PromptTemplate(
    input_variables=["datos_csv"],
    template=prompt_analisis.strip(),
)

# ========================
# 4. Función principal
# ========================
def generar_reporte_completo(file_path: str):
    print("🔄 Iniciando análisis del dataset...")
    
    # Paso 1: Análisis técnico del CSV
    analisis_tecnico = analizar_csv(file_path)
    print("✅ Análisis técnico completado")
    
    # Paso 2: Enviar al LLM para análisis avanzado
    print("🧠 Generando insights con IA...")
    
    prompt_final = prompt.format(datos_csv=analisis_tecnico)
    respuesta = llm.invoke(prompt_final)
    
    # Paso 3: Mostrar resultados
    print("\n" + "="*70)
    print("📊 INFORME COMPLETO DEL DATASET")
    print("="*70)
    print("\n📋 **ANÁLISIS TÉCNICO:**")
    print(analisis_tecnico)
    print("\n" + "="*70)
    print("🤖 **ANÁLISIS AVANZADO CON IA:**")
    print(respuesta.content)
    print("="*70)

# ========================
# 5. Ejecución
# ========================
if __name__ == "__main__":
    # Cambia esta ruta por tu archivo CSV
    archivo_csv = "tu_archivo.csv"  # ← MODIFICA ESTA LÍNEA
    
    if not os.path.exists(archivo_csv):
        print(f"❌ El archivo '{archivo_csv}' no existe.")
        print("Por favor, modifica la variable 'archivo_csv' con la ruta correcta")
    else:
        generar_reporte_completo(archivo_csv)