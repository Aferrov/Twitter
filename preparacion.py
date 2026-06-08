import pandas as pd
import torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# =====================================================================
# 1. CONFIGURACION DE PARAMETROS Y ENTORNO
# =====================================================================
# Tamano del lote para el procesamiento de la traduccion
TAMANO_LOTE = 16 
archivo_salida = "humaid_es.json"

# Configurar el uso de tarjeta grafica si esta disponible
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de ejecucion: {dispositivo.upper()}")

# =====================================================================
# 2. CARGA DE MODELOS Y CONJUNTOS DE DATOS
# =====================================================================
print("Cargando el modelo de traduccion...")
tokenizador_traductor = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
modelo_traductor = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es").to(dispositivo)

print("Descargando el conjunto de datos HumAID...")
# Se utiliza verification_mode para evitar validaciones rigidas que causan errores
dataset_crudo = load_dataset("QCRI/HumAID-all", split="train", verification_mode="no_checks")
datos_dataframe = pd.DataFrame(dataset_crudo)
print(f"Total de registros descargados: {len(datos_dataframe)} tweets")

# =====================================================================
# 3. FILTRADO Y CLASIFICACION POR PALABRAS CLAVE
# =====================================================================
print("Filtrando categorias de interes...")

# Definir la lista de categorias validas para el estudio
clases_validas = [
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "displaced_people_and_evacuations",
    "rescue_volunteering_or_donation_efforts",
    "not_humanitarian",
    "other_relevant_information",
    "sympathy_and_support",
]
datos_filtrados = datos_dataframe[datos_dataframe["class_label"].isin(clases_validas)].copy()

# Crear la columna para la etiqueta numerica inicial, por defecto en 3 (Otros)
datos_filtrados["etiqueta"] = 3

# Clasificar como Sismo (0) si contiene palabras relacionadas con terremotos
filtro_sismo = datos_filtrados["tweet_text"].str.lower().str.contains("earthquake|quake|tremor|seismic", na=False)
datos_filtrados.loc[filtro_sismo, "etiqueta"] = 0

# Clasificar como Lluvia (1) si contiene palabras relacionadas con inundaciones
filtro_lluvia = datos_filtrados["tweet_text"].str.lower().str.contains("flood|rain|hurricane|typhoon|cyclone", na=False)
datos_filtrados.loc[filtro_lluvia, "etiqueta"] = 1

# Forzar la etiqueta Otros (3) para categorias informativas generales o de apoyo
filtro_otros = datos_filtrados["class_label"].isin(["not_humanitarian", "other_relevant_information", "sympathy_and_support"])
datos_filtrados.loc[filtro_otros, "etiqueta"] = 3

# Remover registros sin texto o sin etiqueta valida
datos_limpios = datos_filtrados.dropna(subset=["etiqueta", "tweet_text"]).copy()
datos_limpios["etiqueta"] = datos_limpios["etiqueta"].astype(int)

print(f"Registros despues del filtrado: {len(datos_limpios)}")
print(datos_limpios["etiqueta"].value_counts())

# =====================================================================
# 4. PROCESO DE TRADUCCION AL ESPAÑOL
# =====================================================================
print("Iniciando la traduccion de los textos...")

def ejecutar_traduccion(lista_textos):
    textos_traducidos = []
    
    # Procesar los textos en bloques segun el tamano del lote definido
    for posicion in tqdm(range(0, len(lista_textos), TAMANO_LOTE), desc="Progreso"):
        bloque_textos = [f">>es<< {str(texto)}" for texto in lista_textos[posicion : posicion + TAMANO_LOTE]]
        
        # Convertir los textos en tokens compatibles con el modelo
        tokens = tokenizador_traductor(
            bloque_textos, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(dispositivo)
        
        # Generar las traducciones sin calcular gradientes para ahorrar memoria
        with torch.no_grad():
            identificadores = modelo_traductor.generate(**tokens)
            
        bloque_traducido = tokenizador_traductor.batch_decode(identificadores, skip_special_tokens=True)
        textos_traducidos.extend(bloque_traducido)
        
    return textos_traducidos

# Ejecutar la funcion sobre la columna de texto original
datos_limpios["texto"] = ejecutar_traduccion(datos_limpios["tweet_text"].tolist())

# =====================================================================
# 5. ALMACENAMIENTO DE RESULTADOS
# =====================================================================
# Seleccionar unicamente las columnas requeridas para el entrenamiento posterior
datos_finales = datos_limpios[["texto", "etiqueta"]].copy()
datos_finales.to_json(archivo_salida, orient="records", force_ascii=False, indent=2)

print(f"Proceso completado. Datos guardados en: '{archivo_salida}'")
print(f"Total de filas generadas: {len(datos_finales)}")

print("\nMuestra de registros por cada etiqueta:")
for num_etiqueta, nombre_clase in {0: "sismo", 1: "lluvia", 3: "otros"}.items():
    print(f"\n  Categoria: {nombre_clase.upper()}")
    muestras = datos_finales[datos_finales["etiqueta"] == num_etiqueta]["texto"].head(2)
    for fila_texto in muestras:
        print(f"    -> {fila_texto}")