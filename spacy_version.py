import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import joblib
import spacy
from transformers import pipeline
from scipy import sparse  # Para guardar la matriz en formato disperso
import torch

#########################################
# Cargar el modelo de spaCy
#########################################
try:
    nlp = spacy.load("en_core_web_lg")
    print("Modelo spaCy 'en_core_web_lg' cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo spaCy: {e}")

#########################################
# Precompilación de expresiones regulares
#########################################
URL_PATTERN = re.compile(r'http\S+|www\.\S+')
DIGIT_PATTERN = re.compile(r'\d+')
NON_ALPHA_PATTERN = re.compile(r'[^a-z\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')

#########################################
# Función de limpieza y procesamiento de texto con spaCy
#########################################
def limpiar_texto(texto):
    try:
        texto = str(texto)
        # Elimina caracteres no ASCII
        texto = texto.encode('ascii', 'ignore').decode('ascii')
        texto = texto.lower()
        texto = URL_PATTERN.sub(' ', texto)
        texto = DIGIT_PATTERN.sub(' ', texto)
        texto = NON_ALPHA_PATTERN.sub(' ', texto)
        texto = WHITESPACE_PATTERN.sub(' ', texto).strip()
        return texto if len(texto) > 3 else ''
    except Exception as e:
        print(f"Error en limpiar_texto: {e}")
        return ''

def procesar_texto(texto):
    """
    Limpia y procesa el texto utilizando spaCy:
      - Elimina URLs, dígitos y caracteres no alfabéticos.
      - Convierte a minúsculas.
      - Procesa con spaCy para tokenización, lematización y eliminación de stop words.
    """
    try:
        texto = limpiar_texto(texto)
        doc = nlp(texto)
        # Seleccionar tokens: solo palabras alfabéticas, sin stop words y de longitud > 3
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token.text) > 3]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error al procesar texto: {e}")
        return ''

#########################################
# Carga de datos y generación del corpus
#########################################
def cargar_datos(input_excel):
    try:
        hojas = pd.read_excel(input_excel, sheet_name=None, header=None)
        textos, etiquetas, datos_completos = [], [], []
        for nombre_hoja, data in hojas.items():
            encabezados = data.iloc[0].tolist()
            data.columns = encabezados
            data = data.iloc[1:]
            for _, row in data.iterrows():
                if "text" in row and pd.notna(row["text"]):
                    texto_original = str(row["text"])
                    texto_procesado = procesar_texto(texto_original)
                    textos.append(texto_procesado)
                    etiquetas.append(f"documento{len(textos)}")
                    datos_completos.append({
                        "hoja": nombre_hoja,
                        "documento_id": f"documento{len(textos)}",
                        "texto_original": texto_original,
                        "texto_procesado": texto_procesado
                    })
        return textos, etiquetas, datos_completos
    except Exception as e:
        print(f"Error al cargar datos: {e}")

def guardar_corpus_limpio(datos_completos, ruta_salida):
    try:
        df = pd.DataFrame(datos_completos)
        df.to_excel(ruta_salida, index=False)
        print(f"Corpus limpio guardado en: {ruta_salida}")
    except Exception as e:
        print(f"Error al guardar corpus limpio: {e}")

#########################################
# TF-IDF y funciones auxiliares
#########################################
def construir_matriz_tfidf(textos, ngram_range):
    try:
        vectorizador = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=ngram_range)
        X_tfidf = vectorizador.fit_transform(textos)
        vocabulario = vectorizador.get_feature_names_out()
        print(f"Vocabulario {ngram_range}: {len(vocabulario)} términos")
        return X_tfidf, vocabulario, vectorizador
    except Exception as e:
        print(f"Error al construir matriz TF-IDF: {e}")
        return None, [], None

def guardar_matriz_sparse(X_tfidf, ruta_salida):
    """
    Guarda la matriz TF-IDF en formato disperso (.npz) usando scipy.
    """
    try:
        sparse.save_npz(ruta_salida, X_tfidf)
        print(f"Matriz TF-IDF (sparse) guardada en: {ruta_salida}")
    except Exception as e:
        print(f"Error al guardar la matriz TF-IDF sparse: {e}")

def guardar_diccionario_csv(textos, vocabulario, ruta_salida):
    try:
        frecuencia_palabras = Counter()
        filas_palabras = {}
        for i, texto in enumerate(textos):
            tokens = texto.split()
            frecuencia_palabras.update(tokens)
            for token in tokens:
                filas_palabras.setdefault(token, set()).add(i + 1)
        datos_diccionario = [
            [palabra, ','.join(map(str, sorted(filas_palabras.get(palabra, [])))), frecuencia_palabras.get(palabra, 0)]
            for palabra in vocabulario
        ]
        df_diccionario = pd.DataFrame(datos_diccionario, columns=["Palabra", "Filas", "Frecuencia"])
        df_diccionario.to_csv(ruta_salida, index=False)
        print(f"Diccionario guardado en: {ruta_salida}")
    except Exception as e:
        print(f"Error al guardar diccionario: {e}")

#########################################
# LSA utilizando GPU (SVD con PyTorch)
#########################################
def aplicar_lsa_gpu(X_tfidf, n_componentes):
    try:
        # Convertir la matriz dispersa a densa (¡cuidado con la memoria!)
        dense = X_tfidf.toarray()
        tensor = torch.tensor(dense, dtype=torch.float32, device='cuda')
        U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
        U_trunc = U[:, :n_componentes]
        S_trunc = S[:n_componentes]
        X_lsa = U_trunc * S_trunc.unsqueeze(0)
        X_lsa_cpu = X_lsa.cpu().numpy()
        print("LSA aplicado en GPU.")
        return X_lsa_cpu, S_trunc.cpu().numpy(), Vh[:n_componentes].cpu().numpy()
    except Exception as e:
        print(f"Error en aplicar_lsa_gpu: {e}")
        return None, None, None

def copiar_texto_procesado(input_excel, output_excel):
    try:
        df = pd.read_excel(input_excel)
        if 'texto_procesado' in df.columns:
            df[['texto_procesado']].to_excel(output_excel, index=False)
            print(f"Columna 'texto_procesado' copiada a: {output_excel}")
        else:
            print("No se encontró la columna 'texto_procesado'.")
    except Exception as e:
        print(f"Error al copiar texto procesado: {e}")

#########################################
# Análisis de sentimientos con GPU
#########################################
try:
    sentiment_pipeline = pipeline("sentiment-analysis", 
                                  model="cardiffnlp/twitter-roberta-base-sentiment", 
                                  device=0)
    SENTIMENT_MAPPING = {"LABEL_0": "Negativa", "LABEL_1": "Neutral", "LABEL_2": "Positiva"}
    print("Pipeline de sentimiento inicializado en GPU.")
except Exception as e:
    print(f"Error al inicializar el pipeline de sentimiento: {e}")
    sentiment_pipeline = None
    SENTIMENT_MAPPING = {}

def analizar_sentimientos_corpus(ruta_archivo, batch_size=32):
    try:
        df = pd.read_excel(ruta_archivo)
        if 'texto_procesado' not in df.columns:
            print(f"Error: No se encontró la columna 'texto_procesado' en {ruta_archivo}")
            return
        
        textos = df['texto_procesado'].fillna("").tolist()
        if sentiment_pipeline is None:
            print("Pipeline de sentimiento no disponible.")
            return
        
        resultados = sentiment_pipeline(textos, batch_size=batch_size)
        sentimientos, polaridades = [], []
        for res in resultados:
            label = res.get('label', 'LABEL_1')
            score = res.get('score', 0.0)
            sentiment = SENTIMENT_MAPPING.get(label, "Neutral")
            polaridad = score if sentiment == "Positiva" else (-score if sentiment == "Negativa" else 0.0)
            sentimientos.append(sentiment)
            polaridades.append(polaridad)
        
        df['sentimiento'] = sentimientos
        df['polaridad'] = polaridades
        
        ruta_salida = ruta_archivo.replace('.xlsx', '_con_sentimiento.xlsx')
        df.to_excel(ruta_salida, index=False)
        print(f"Análisis de sentimientos guardado en: {ruta_salida}")
        
        conteo = df['sentimiento'].value_counts()
        plt.figure(figsize=(8,6))
        colores = {'Positiva': 'green', 'Neutral': 'gray', 'Negativa': 'red'}
        conteo.plot(kind='bar', color=[colores.get(x, 'blue') for x in conteo.index])
        plt.title('Distribución de Sentimientos')
        plt.xlabel('Sentimiento')
        plt.ylabel('Cantidad')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        ruta_grafico = os.path.join(os.path.dirname(ruta_archivo), 'distribucion_sentimientos.png')
        plt.savefig(ruta_grafico)
        print(f"Gráfico de distribución guardado en: {ruta_grafico}")
    except Exception as e:
        print(f"Error al analizar sentimientos: {e}")

#########################################
# Función principal
#########################################
def main():
    try:
        inicio = time.time()
        # Asegúrate de tener el archivo 'corpus_millon.xlsx' en el directorio actual
        input_excel = "corpus_millon.xlsx"
        os.makedirs("lsa_simple", exist_ok=True)
    
        textos, etiquetas, datos_completos = cargar_datos(input_excel)
        guardar_corpus_limpio(datos_completos, "lsa_simple/corpus_limpio.xlsx")
        copiar_texto_procesado("lsa_simple/corpus_limpio.xlsx", "lsa_simple/corpus_clasificacion.xlsx")
    
        for ngram in [(1,1), (2,2), (3,3)]:
            X_tfidf, vocabulario, _ = construir_matriz_tfidf(textos, ngram)
            sufijo = { (1,1): "", (2,2): "_bigramas", (3,3): "_trigramas" }[ngram]
            ruta_npz = f"lsa_simple/matriz{sufijo}.npz"
            guardar_matriz_sparse(X_tfidf, ruta_npz)
            guardar_diccionario_csv(textos, vocabulario, f"lsa_simple/diccionario{sufijo}.csv")
    
        X_lsa, S_trunc, Vh_trunc = aplicar_lsa_gpu(X_tfidf, 50)
    
        print("\nIniciando análisis de sentimientos...")
        analizar_sentimientos_corpus("lsa_simple/corpus_clasificacion.xlsx", batch_size=32)
    
        fin = time.time()
        print(f"\nTiempo total de ejecución: {fin - inicio:.2f} segundos.")
    except Exception as e:
        print(f"Error en el proceso principal: {e}")

if __name__ == "__main__":
    main()