import pandas as pd
import numpy as np
import os
import nltk
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import threading
import time


def descargar_recursos_nltk():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    except Exception as e:
        print(f"Error al descargar recursos NLTK: {e}")

def limpiar_texto(texto):
    try:
        texto = str(texto)
        texto = texto.encode('ascii', 'ignore').decode('ascii')
        texto = texto.lower()
        texto = re.sub(r'http\S+|www\.\S+', ' ', texto)
        texto = re.sub(r'\d+', ' ', texto)
        texto = re.sub(r'[^a-z\s]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto)
        texto = texto.strip()
        return texto if len(texto) > 3 else ''
    except Exception as e:
        print(f"Error al descargar recursos NLTK: {e}")
        return ''
    

def tokenizar_texto(texto):
    try:
        tokenizer = RegexpTokenizer(r'\w+')
        return tokenizer.tokenize(texto)
    except Exception as e:
        print(f"Error al tokenizar texto: {e}")
        return []


def eliminar_stopwords(tokens):
    try:
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token not in stop_words and len(token) > 3]
    except Exception as e:
        print(f"Error al eliminar stopwords texto: {e}")
        return tokens



def lematizar(tokens):
    try:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    except Exception as e:
        print(f"Error al lematizar texto: {e}")
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


def cargar_datos(input_excel):
    try:
        hojas = pd.read_excel(input_excel, sheet_name=None, header=None)
        textos, etiquetas = [], []
        textos_originales = []  # Para guardar los textos originales
        textos_procesados = []  # Para guardar los textos procesados
        datos_completos = []    # Para guardar todos los datos para el corpus limpio
    
        for nombre_hoja, data in hojas.items():
            encabezados = data.iloc[0].tolist()
            data.columns = encabezados
            data = data.iloc[1:]
            for index, row in data.iterrows():
                if "text" in row and pd.notna(row["text"]):
                    texto_original = str(row["text"])
                    texto_procesado = procesar_texto(texto_original)
                    textos.append(texto_procesado)
                    etiquetas.append(f"documento{len(textos)}")
                    textos_originales.append(texto_original)
                    textos_procesados.append(texto_procesado)
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
    """
    Guarda los textos originales y procesados en un nuevo archivo Excel.
    """
    try:
        df = pd.DataFrame(datos_completos)
        df.to_excel(ruta_salida, index=False)
        print(f"Corpus limpio guardado en: {ruta_salida}")
    except Exception as e:
        print(f"Error al guardar corpus limpio: {e}")
       
    

def construir_matriz_tfidf(textos, ngram_range):
    try:
        vectorizador = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=ngram_range)
        X_tfidf = vectorizador.fit_transform(textos)
        vocabulario = vectorizador.get_feature_names_out()
        print(f"Número de términos en el vocabulario {ngram_range}: {len(vocabulario)}")
        return X_tfidf, vocabulario, vectorizador
    except Exception as e:
        print(f"Error al construir matriz TF-TDF: {e}")
        return None, [], None

def guardar_matriz_csv(X_tfidf, vocabulario, etiquetas, ruta_salida):
    try:
        df_matriz = pd.DataFrame(X_tfidf.toarray(), index=etiquetas, columns=vocabulario)
        df_matriz.to_csv(ruta_salida)
        print(f"Matriz TF-IDF guardada en: {ruta_salida}")
    except Exception as e:
        print(f"Error al guardar la matriz TF-TDF: {e}")
        

def guardar_diccionario_csv(textos, vocabulario, ruta_salida):
    try:
        frecuencia_palabras = Counter()
        filas_palabras = {}
    
        for i, texto in enumerate(textos):
            tokens = texto.split()
            frecuencia_palabras.update(tokens)
            for token in tokens:
                if token not in filas_palabras:
                    filas_palabras[token] = []
                filas_palabras[token].append(i + 1)  # Número de fila (1-based)
    
        datos_diccionario = []
        for palabra in vocabulario:
            datos_diccionario.append([palabra, ', '.join(map(str, set(filas_palabras.get(palabra, [])))), frecuencia_palabras.get(palabra, 0)])
    
        df_diccionario = pd.DataFrame(datos_diccionario, columns=["Palabra", "Filas", "Frecuencia"])
        df_diccionario.to_csv(ruta_salida, index=False)
        print(f"Diccionario guardado en: {ruta_salida}")
    except Exception as e:
        print(f"Error al guardar diccionario: {e}")
        return ''

def aplicar_lsa(X_tfidf, n_componentes):
    try:
        svd = TruncatedSVD(n_components=n_componentes)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X_lsa = lsa.fit_transform(X_tfidf)
        return X_lsa, svd, lsa
    except Exception as e:
        print(f"Error al aplicar LSA: {e}")
        return ''


def copiar_texto_procesado(input_excel, output_excel):
    """
    Copia la columna 'texto_procesado' del archivo input_excel a un nuevo archivo output_excel.
    """
    try:
        df = pd.read_excel(input_excel)
        df_texto_procesado = df[['texto_procesado']]
        df_texto_procesado.to_excel(output_excel, index=False)
        print(f"Columna 'texto_procesado' copiada a: {output_excel}")
    except Exception as e:
        print(f"Error al copiar texto procesado: {e}")
        return ''

def traducir_texto(texto, destino='en'):
    try:
        # Asegurarse de que el texto sea un string
        texto = str(texto) if texto is not None else ""
        if texto.strip() == "":
            return ""
        
        traductor = Translator()
        traduccion = traductor.translate(texto, dest=destino)
        return traduccion.text
    except Exception as e:
        print(f"Error en traducción: {e}")
        return texto  # En caso de error, devolver el texto original

def clasificar_opinion(texto):
    try:
        texto = str(texto) if texto is not None else ""
        if not texto or texto.strip() == "":
            return "Neutral", 0
    
        analyzer = SentimentIntensityAnalyzer()
        resultados = analyzer.polarity_scores(texto)
        polaridad = resultados['compound']
    
        if polaridad < -0.05:
            clasificacion = "Negativa"
        elif polaridad > 0.05:
            clasificacion = "Positiva"
        else:
            clasificacion = "Neutral"
    
        return clasificacion, polaridad
    except Exception as e:
        print(f"Error en clasificar opinion: {e}")
        return texto  # En caso de error, devolver el texto original

def procesar_fragmento(df, inicio, fin, resultados, lock):
    try:
        for idx in range(inicio, fin):
            texto = str(df.at[idx, 'texto_procesado']) if pd.notna(df.at[idx, 'texto_procesado']) else ""
            sentimiento, polaridad = clasificar_opinion(texto)
        
            with lock:
                resultados[idx] = (sentimiento, polaridad)
    except Exception as e:
        print(f"Error al procesar fragmento: {e}")
        return texto  # En caso de error, devolver el texto original

def analizar_sentimientos_corpus(ruta_archivo, num_hilos=4):
    try:
        df = pd.read_excel(ruta_archivo)
        if 'texto_procesado' not in df.columns:
            print(f"Error: No se encontró la columna 'texto_procesado' en {ruta_archivo}")
            return
        
        df['sentimiento'] = ""
        df['polaridad'] = 0.0
        
        resultados = {}
        lock = threading.Lock()
        hilos = []
        chunk_size = len(df) // num_hilos
        
        for i in range(num_hilos):
            inicio = i * chunk_size
            fin = (i + 1) * chunk_size if i != num_hilos - 1 else len(df)
            hilo = threading.Thread(target=procesar_fragmento, args=(df, inicio, fin, resultados, lock))
            hilos.append(hilo)
            hilo.start()
        
        for hilo in hilos:
            hilo.join()
        
        for idx, (sentimiento, polaridad) in resultados.items():
            df.at[idx, 'sentimiento'] = sentimiento
            df.at[idx, 'polaridad'] = polaridad
        
        ruta_salida = ruta_archivo.replace('.xlsx', '_con_sentimiento.xlsx')
        df.to_excel(ruta_salida, index=False)
        print(f"Análisis de sentimientos guardado en: {ruta_salida}")
        
        conteo_sentimientos = df['sentimiento'].value_counts()
        plt.figure(figsize=(8, 6))
        colores = {'Positiva': 'green', 'Neutral': 'gray', 'Negativa': 'red'}
        conteo_sentimientos.plot(kind='bar', color=[colores.get(x, 'blue') for x in conteo_sentimientos.index])
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


def main():
    try:
        inicio = time.time()

        descargar_recursos_nltk()
        input_excel = "corpus_millon.xlsx"
        os.makedirs("lsa_simple", exist_ok=True)
    
        textos, etiquetas, datos_completos = cargar_datos(input_excel)
    
        # Guardar corpus limpio
        guardar_corpus_limpio(datos_completos, "lsa_simple/corpus_limpio.xlsx")
    
        # Copiar columna 'texto_procesado' a un nuevo archivo
        copiar_texto_procesado("lsa_simple/corpus_limpio.xlsx", "lsa_simple/corpus_clasificacion.xlsx")
    
        # Unigramas
        X_tfidf, vocabulario, _ = construir_matriz_tfidf(textos, (1,1))
        guardar_matriz_csv(X_tfidf, vocabulario, etiquetas, "lsa_simple/matriz.csv")
        guardar_diccionario_csv(textos, vocabulario, "lsa_simple/diccionario.csv")
    
        # Bigramas
        X_tfidf_bigramas, vocabulario_bigramas, _ = construir_matriz_tfidf(textos, (2,2))
        guardar_matriz_csv(X_tfidf_bigramas, vocabulario_bigramas, etiquetas, "lsa_simple/matriz_bigramas.csv")
        guardar_diccionario_csv(textos, vocabulario_bigramas, "lsa_simple/diccionario_bigramas.csv")
    
        # Trigramas
        X_tfidf_trigramas, vocabulario_trigramas, _ = construir_matriz_tfidf(textos, (3,3))
        guardar_matriz_csv(X_tfidf_trigramas, vocabulario_trigramas, etiquetas, "lsa_simple/matriz_trigramas.csv")
        guardar_diccionario_csv(textos, vocabulario_trigramas, "lsa_simple/diccionario_trigramas.csv")
    
        # LSA
        X_lsa, svd, lsa = aplicar_lsa(X_tfidf, 50)
        #visualizar_lsa(X_lsa, etiquetas, "lsa_simple/lsa_visualizacion.png")
    
        # Análisis de sentimientos en el corpus de clasificación
        print("\nIniciando análisis de sentimientos en el corpus de clasificación...")
        analizar_sentimientos_corpus("lsa_simple/corpus_clasificacion.xlsx")
    
        print("\nProceso completo. Se ha añadido el análisis de sentimientos al corpus de clasificación.")

     
        fin = time.time()
        print(f"\nProceso completo. Tiempo total de ejecución: {fin - inicio:.2f} segundos.")

    except Exception as e:
        print(f"Error el proceso principal: {e}")


if __name__ == "__main__":
    main()