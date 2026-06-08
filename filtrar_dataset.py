import urllib.request
import csv
import random

# Forzar una semilla para que el desorden aleatorio sea siempre el mismo
random.seed(42)

# Enlace base del repositorio para descargar los tweets reales en espanol
ENLACE_BASE = "https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/spanish/"

# Lista de archivos de texto y etiquetas para descargar
ARCHIVOS = {
    "train": ("train_text.txt", "train_labels.txt"),
    "val":   ("val_text.txt",   "val_labels.txt"),
    "test":  ("test_text.txt",  "test_labels.txt"),
}

# Lista de nombres de usuarios locales simulados
USUARIOS = [
    "carlos_arequipa", "maria_misti", "pedro_volcan", "lucia_sillar",
    "jose_chili", "ana_cayma", "roberto_yanahuara", "sofia_mollendo",
    "miguel_sabandia", "elena_socabaya", "diego_cerro", "rosa_tiabaya",
    "juan_sachaca", "carmen_paucarpata", "luis_mariano", "patricia_flores",
    "andres_quispe", "natalia_bustamante", "marco_fernandez", "diana_mamani",
]

# Palabras clave para la clasificacion inicial de los textos
PALABRAS_CLAVE = {
    "sismo":   ["sismo", "terremoto", "temblor", "réplica", "seísmo", "epicentro", "sacudió", "magnitud", "tectónico"],
    "lluvia":  ["huayco", "aluvión", "inundación", "desborde", "lluvia", "aniego", "torrencial", "crecida", "granizada"],
    "trafico": ["accidente", "choque", "congestión", "tráfico", "volcado", "atropello", "desvío", "colisión", "semáforo"],
}

# Funcion para descargar los archivos desde internet
def descargar_archivo(url):
    print(f"Descargando {url.split('/')[-1]}...")
    with urllib.request.urlopen(url) as respuesta:
        return respuesta.read().decode("utf-8").strip().splitlines()

# Lista para acumular todas las filas generadas
lista_filas = []

# Procesar y combinar los textos con sus etiquetas
for division, (archivo_txt, archivo_lbl) in ARCHIVOS.items():
    print(f"\nProcesando division: {division.upper()}")
    try:
        textos = descargar_archivo(ENLACE_BASE + archivo_txt)
        etiquetas = descargar_archivo(ENLACE_BASE + archivo_lbl)

        for texto, etiqueta in zip(textos, etiquetas):
            texto_limpio = texto.strip()
            
            # Omitir textos demasiado cortos o vacios
            if len(texto_limpio) < 5:
                continue

            texto_minusculas = texto_limpio.lower()

            # --- Reglas de Negocio: Estado Emocional ---
            if any(p in texto_minusculas for p in ["ayuda", "auxilio", "socorro", "atrapados", "heridos", "s.o.s", "sos"]):
                estado_detectado = "Ayuda"
            elif any(p in texto_minusculas for p in ["pánico", "panico", "miedo", "terror", "desesperante", "rezos", "dios mío", "dios mio"]):
                estado_detectado = "Pánico"
            elif any(p in texto_minusculas for p in ["alcalde", "municipio", "incompetentes", "desgracia", "culpable", "brilla por su ausencia"]):
                estado_detectado = "Denuncia"
            else:
                estado_detectado = "Informativo"

            # --- Reglas de Negocio: Prioridad Operativa ---
            if any(p in texto_minusculas for p in ["urgente", "ahora", "ya", "inmediato", "morir", "ahogar", "colapso", "destruido"]):
                prioridad_detectada = "Urgente"
            else:
                prioridad_detectada = "Normal"

            # Agregar el registro estructurado a la lista
            lista_filas.append({
                "text": texto_limpio,
                "user": random.choice(USUARIOS),
                "estado": estado_detectado,
                "prioridad": prioridad_detectada
            })
            
        print(f"Completado: {len(textos)} registros procesados")
    except Exception as error:
        print(f"Error en la descarga o procesamiento: {error}")

# Desordenar la lista final de registros de manera aleatoria
random.shuffle(lista_filas)

# Guardar los datos en el archivo CSV para las pruebas de la capa batch
archivo_salida = "dataset_pruebas_expertas.csv"
columnas = ["text", "user", "estado", "prioridad"]

with open(archivo_salida, "w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(f, fieldnames=columnas)
    escritor.writeheader()
    escritor.writerows(lista_filas)

# Mostrar resumen del archivo guardado en la consola
from collections import Counter
print(f"\n=======================================================")
print(f"Archivo guardado exitosamente: '{archivo_salida}'")
print(f"Total de registros guardados: {len(lista_filas)}")

print(f"\nDistribucion por Estado Emocional:")
for clave, valor in Counter(r["estado"] for r in lista_filas).most_common():
    barra_visual = "█" * (valor // 10)
    print(f"  {clave:<12}: {valor:>4}  {barra_visual}")

print(f"\nDistribucion por Prioridad Operativa:")
for clave, valor in Counter(r["prioridad"] for r in lista_filas).most_common():
    barra_visual = "█" * (valor // 10)
    print(f"  {clave:<12}: {valor:>4}  {barra_visual}")

print(f"\nMuestra de los primeros 2 registros:")
for registro in lista_filas[:2]:
    print(f"  -> Texto: {registro['text'][:70]}...")
    print(f"     user={registro['user']} | estado={registro['estado']} | prioridad={registro['prioridad']}\n")