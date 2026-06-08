import os
import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# =====================================================================
# 1. CONFIGURACION DE RUTA Y HARDWARE
# =====================================================================
# Detectar si la tarjeta grafica esta disponible para acelerar el proceso
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
modelo_base = "dccuchile/bert-base-spanish-wwm-cased"

print(f"Iniciando entrenamiento del modelo en: {dispositivo.upper()}")

# =====================================================================
# 2. FUNCION DE LIMPIEZA DE TEXTO
# =====================================================================
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    
    # Pasar a minusculas y remover menciones, enlaces y el caracter hashtag
    texto = texto.lower()
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'http\s+|https\S+', '', texto)
    texto = texto.replace('#', '')
    
    # Reemplazar tildes y caracteres especiales
    texto = texto.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    texto = re.sub(r'[^\w\s,.:;¡!¿?]', '', texto)
    
    # Remover espacios en blanco duplicados
    return " ".join(texto.split())

# =====================================================================
# 3. CARGA Y PREPARACION DE LOS DATOS TRADUCIDOS
# =====================================================================
if __name__ == "__main__":
    print("Buscando archivo de datos: 'humaid_es.json'...")
    
    # Validar si el archivo de datos existe en el directorio
    if os.path.exists("humaid_es.json"):
        datos_json = pd.read_json("humaid_es.json")
        
        # Renombrar las columnas para cumplir con el estandar de la libreria
        datos_dataframe = datos_json.rename(columns={"texto": "text", "etiqueta": "label"})
        print(f"Archivo cargado correctamente con {len(datos_dataframe)} filas originales.")
        
        # Tomar una muestra maxima por cada clase para agilizar el procesamiento
        datos_dataframe = datos_dataframe.groupby("label").sample(
            n=min(600, len(datos_dataframe)), random_state=42
        ).reset_index(drop=True)
    else:
        print("Error: No se encontro el archivo 'humaid_es.json'.")
        print("Asegurate de haber ejecutado primero el archivo de descarga y traduccion.")
        exit()

    # Aplicar la funcion de limpieza de texto sobre la columna de interes
    datos_dataframe["text"] = datos_dataframe["text"].apply(limpiar_texto)
    datos_dataframe = datos_dataframe.dropna(subset=["text", "label"])
    datos_dataframe["label"] = datos_dataframe["label"].astype(int)
    
    print(f"Total de registros seleccionados para entrenar: {len(datos_dataframe)}")
    print(datos_dataframe["label"].value_counts())

    # =====================================================================
    # 4. DIVISION DEL CONJUNTO DE DATOS (80% Entrenamiento / 20% Validacion)
    # =====================================================================
    datos_train, datos_val = train_test_split(datos_dataframe, test_size=0.2, random_state=42)
    
    # Convertir los conjuntos de pandas al formato nativo Dataset
    dataset_train = Dataset.from_pandas(datos_train.reset_index(drop=True))
    dataset_val = Dataset.from_pandas(datos_val.reset_index(drop=True))

    # =====================================================================
    # 5. PROCESO DE TOKENIZACION
    # =====================================================================
    print("Cargando el tokenizador del modelo base...")
    tokenizador = AutoTokenizer.from_pretrained(modelo_base)

    def aplicar_tokenizacion(ejemplos):
        return tokenizador(ejemplos["text"], truncation=True, padding="max_length", max_length=128)

    print("Convirtiendo los textos en secuencias numericas...")
    dataset_train = dataset_train.map(aplicar_tokenizacion, batched=True)
    dataset_val = dataset_val.map(aplicar_tokenizacion, batched=True)

    # =====================================================================
    # 6. CONFIGURACION DEL MODELO Y LOS ARGUMENTOS DE ENTRENAMIENTO
    # =====================================================================
    print("Configurando la arquitectura para clasificacion de 4 neuronas de salida...")
    modelo = AutoModelForSequenceClassification.from_pretrained(modelo_base, num_labels=4)

    # Definir los parametros de entrenamiento optimizados para el entorno
    config_entrenamiento = TrainingArguments(
        output_dir="./resultados_entrenamiento",
        eval_strategy="epoch",        # Evaluar al finalizar cada ciclo completo
        save_strategy="no",           # No guardar archivos intermedios para ahorrar espacio
        learning_rate=2e-5,           # Tasa de aprendizaje estándar para ajuste fino
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,           # Cantidad de ciclos de entrenamiento sobre el conjunto completo
        weight_decay=0.01,            # Parametro para mitigar el sobreajuste
        logging_steps=10,
        disable_tqdm=False
    )

    # Inicializar la herramienta de entrenamiento de Hugging Face
    entrenador = Trainer(
        model=modelo,
        args=config_entrenamiento,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
    )

    # =====================================================================
    # 7. EJECUCION DEL ENTRENAMIENTO Y ALMACENAMIENTO DEL MODELO FINAL
    # =====================================================================
    print("Iniciando los ciclos de ajuste fino en el modelo...")
    entrenador.train()
    
    ruta_salida = "./modelo_alertas_arequipa"
    print(f"Guardando el modelo entrenado y su tokenizador en: '{ruta_salida}'")
    modelo.save_pretrained(ruta_salida)
    tokenizador.save_pretrained(ruta_salida)
    
    print("Proceso finalizado correctamente.")