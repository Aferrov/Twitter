import os
import pandas as pd

# =====================================================================
# 1. CONFIGURACION DE ARCHIVOS
# =====================================================================
archivo_entrada = "humaid_es.json"
archivo_salida = "dataset_pruebas_expertas.csv"

print("Iniciando compilador de archivos de texto")

if __name__ == "__main__":
    
    print(f"Buscando archivo origen local: '{archivo_entrada}'...")
    
    # Validar si el archivo de entrada existe en la carpeta actual
    if not os.path.exists(archivo_entrada):
        print(f"Error: No se encontro el archivo '{archivo_entrada}'.")
        exit()
        
    try:
        # Leer el archivo origen con los tweets en espanol
        datos_json = pd.read_json(archivo_entrada)
        print(f"Archivo origen cargado con {len(datos_json)} registros.")
        
        # Mapear los datos al nuevo formato de estado y prioridad
        datos_nuevos = pd.DataFrame({
            "user": "@humaid_real_user",
            "text": datos_json["texto"]
        })
        
        # Limpieza para quitar registros vacios o duplicados
        datos_nuevos = datos_nuevos.dropna(subset=["text"])
        datos_nuevos = datos_nuevos[datos_nuevos["text"].astype(str).str.strip() != ""]
        datos_nuevos = datos_nuevos.drop_duplicates(subset=["text"])
        
        # Seleccionar una muestra maxima de 3000 registros para las pruebas
        total_registros = min(3000, len(datos_nuevos))
        datos_muestra = datos_nuevos.sample(n=total_registros, random_state=42).reset_index(drop=True)
        
        # Crear las columnas de estado y prioridad segun el contenido del texto
        lista_estados = []
        lista_prioridades = []
        
        for texto in datos_muestra["text"]:
            texto_minusculas = str(texto).lower()
            
            # Asignar estado segun palabras clave
            if any(p in texto_minusculas for p in ["ayuda", "auxilio", "socorro", "atrapados", "heridos", "s.o.s", "sos"]):
                estado = "Ayuda"
            elif any(p in texto_minusculas for p in ["pánico", "panico", "miedo", "terror", "desesperante", "rezos", "dios mío", "dios mio"]):
                estado = "Pánico"
            elif any(p in texto_minusculas for p in ["alcalde", "municipio", "incompetentes", "desgracia", "culpable", "brilla por su ausencia"]):
                estado = "Denuncia"
            else:
                estado = "Informativo"
                
            # Asignar prioridad segun palabras clave
            if any(p in texto_minusculas for p in ["urgente", "ahora", "ya", "inmediato", "morir", "ahogar", "colapso", "destruido"]):
                prioridad = "Urgente"
            else:
                prioridad = "Normal"
                
            lista_estados.append(estado)
            lista_prioridades.append(prioridad)
            
        # Agregar las nuevas columnas determinadas al conjunto final de datos
        datos_muestra["estado"] = lista_estados
        datos_muestra["prioridad"] = lista_prioridades
        
        # Ordenar las columnas para guardarlas en el archivo CSV
        datos_finales = datos_muestra[["text", "user", "estado", "prioridad"]]
        
        # Guardar los datos procesados en el disco
        datos_finales.to_csv(archivo_salida, index=False, encoding="utf-8")
        
        print("Proceso completado exitosamente.")
        print(f"Archivo masivo creado: '{archivo_salida}' con {len(datos_finales)} filas listas.")
        
    except Exception as error:
        print(f"Error al procesar el archivo: {error}")