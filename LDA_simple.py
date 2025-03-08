import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append(r"C:\Users\nesto\Desktop\lsaldaenv\Lib\site-packages")

from imblearn.over_sampling import SMOTE

# Cargar datos
archivo = "lsa_simple/corpus_limpio_con_sentimiento.xlsx"
df = pd.read_excel(archivo)

# Verificar estructura del archivo
if "texto_procesado" not in df.columns or "sentimiento" not in df.columns:
    raise ValueError("El archivo debe contener las columnas 'texto_procesado' y 'sentimiento'.")

# Eliminar valores nulos
df = df.dropna(subset=["texto_procesado", "sentimiento"])

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,1), min_df=2, max_df=0.9)
X = vectorizer.fit_transform(df["texto_procesado"]).toarray()
y = df["sentimiento"].astype(str)

# Imprimir información del vectorizador
print("Número de características:", len(vectorizer.get_feature_names_out()))

# Ver el texto original
print("\nTexto original:")
texto_original = df["texto_procesado"].iloc[0]
print(texto_original)

# Ejemplo de transformación de un documento
vector_ejemplo = vectorizer.transform([texto_original]).toarray()

# Ver características con valores no nulos
indices_no_nulos = np.where(vector_ejemplo[0] != 0)[0]
print("\nCaracterísticas con valores no nulos:")
for idx in indices_no_nulos:
    palabra = vectorizer.get_feature_names_out()[idx]
    valor = vector_ejemplo[0][idx]
    print(f"Palabra: {palabra}, Valor: {valor}")

# Imprimir las primeras 20 características
print("\nCaracterísticas (primeras 20):")
print(vectorizer.get_feature_names_out()[:20])

# Imprimir la matriz vectorizada
print("\nForma de la matriz vectorizada:")
print(X.shape)

# Resto del código de clasificación (continúa como estaba)
# Aplicar SMOTE para balancear clases
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=0.95)  # Mantener el 95% de la varianza
X_pca = pca.fit_transform(X_resampled)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_resampled, test_size=0.2, random_state=42)

# Aplicar LDA con optimización
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

print("\nMatriz de Confusión (LDA):\n", confusion_matrix(y_test, y_pred_lda))
print("\nReporte de Clasificación (LDA):\n", classification_report(y_test, y_pred_lda))

# Prueba con Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred_rf = clf.predict(X_test)

print("\nMatriz de Confusión (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))
print("\nReporte de Clasificación (Random Forest):\n", classification_report(y_test, y_pred_rf))