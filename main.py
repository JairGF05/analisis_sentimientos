import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import torch

#########################################
# Descarga y configuración de NLTK
#########################################
def descargar_recursos_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"Error al descargar recursos NLTK: {e}")

#########################################
# Precompilación de expresiones y objetos globales
#########################################
URL_PATTERN = re.compile(r'http\S+|www\.\S+')
DIGIT_PATTERN = re.compile(r'\d+')
NON_ALPHA_PATTERN = re.compile(r'[^a-z\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')
TOKENIZER = RegexpTokenizer(r'\w+')
STOPWORDS = set(stopwords.words('english'))
LEMATIZER = WordNetLemmatizer()

#########################################
# Funciones de preprocesamiento de texto
#########################################
def limpiar_texto(texto):
    try:
        texto = str(texto)
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

def tokenizar_texto(texto):
    try:
        return TOKENIZER.tokenize(texto)
    except Exception as e:
        print(f"Error al tokenizar texto: {e}")
        return []

def eliminar_stopwords(tokens):
    try:
        return [token for token in tokens if token not in STOPWORDS and len(token) > 3]
    except Exception as e:
        print(f"Error al eliminar stopwords: {e}")
        return tokens

def lematizar(tokens):
    try:
        return [LEMATIZER.lemmatize(token, pos='v') for token in tokens]
    except Exception as e:
        print(f"Error al lematizar: {e}")
        return tokens

def procesar_texto(texto):
    try:
        texto = limpiar_texto(texto)
        tokens = tokenizar_texto(texto)
        tokens = eliminar_stopwords(tokens)
        tokens = lematizar(tokens)
        return ' '.join(tokens)
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

def guardar_matriz_csv(X_tfidf, vocabulario, etiquetas, ruta_salida):
    try:
        df_matriz = pd.DataFrame(X_tfidf.toarray(), index=etiquetas, columns=vocabulario)
        df_matriz.to_csv(ruta_salida)
        print(f"Matriz TF-IDF guardada en: {ruta_salida}")
    except Exception as e:
        print(f"Error al guardar la matriz TF-IDF: {e}")

def guardar_diccionario_csv(textos, vocabulario, ruta_salida):
    try:
        frecuencia_palabras = Counter()
        filas_palabras = {}
        for i, texto in enumerate(textos):
            tokens = texto.split()
            frecuencia_palabras.update(tokens)
            for token in tokens:
                filas_palabras.setdefault(token, set()).add(i+1)
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
        # Convertir a tensor y mover a GPU
        tensor = torch.tensor(dense, dtype=torch.float32, device='cuda')
        # Aplicar SVD
        U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
        # Truncar la SVD
        U_trunc = U[:, :n_componentes]
        S_trunc = S[:n_componentes]
        # Multiplicar U * S para obtener la representación reducida
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
# Inicializamos el pipeline de sentimiento para usar la GPU (device=0)
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
        
        # Procesamiento en batch (utilizando GPU para inferencia)
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
        descargar_recursos_nltk()
        input_excel = "corpus_millon.xlsx"  # Asegúrate de que el archivo tenga el formato esperado
        os.makedirs("lsa_simple", exist_ok=True)
    
        # Cargar y procesar datos
        textos, etiquetas, datos_completos = cargar_datos(input_excel)
        guardar_corpus_limpio(datos_completos, "lsa_simple/corpus_limpio.xlsx")
        copiar_texto_procesado("lsa_simple/corpus_limpio.xlsx", "lsa_simple/corpus_clasificacion.xlsx")
    
        # TF-IDF y diccionarios para unigramas, bigramas y trigramas
        for ngram in [(1,1), (2,2), (3,3)]:
            X_tfidf, vocabulario, _ = construir_matriz_tfidf(textos, ngram)
            sufijo = { (1,1): "", (2,2): "_bigramas", (3,3): "_trigramas" }[ngram]
            guardar_matriz_csv(X_tfidf, vocabulario, etiquetas, f"lsa_simple/matriz{sufijo}.csv")
            guardar_diccionario_csv(textos, vocabulario, f"lsa_simple/diccionario{sufijo}.csv")
    
        # Aplicar LSA en GPU (nota: conversión a densa puede ser intensiva en memoria)
        X_lsa, S_trunc, Vh_trunc = aplicar_lsa_gpu(X_tfidf, 50)
    
        # Análisis de sentimientos en el corpus de clasificación (batch en GPU)
        print("\nIniciando análisis de sentimientos...")
        analizar_sentimientos_corpus("lsa_simple/corpus_clasificacion.xlsx", batch_size=32)
    
        fin = time.time()
        print(f"\nTiempo total de ejecución: {fin - inicio:.2f} segundos.")
    except Exception as e:
        print(f"Error en el proceso principal: {e}")

if __name__ == "__main__":
    main()
