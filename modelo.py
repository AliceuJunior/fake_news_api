import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
from nltk.corpus import stopwords
import nltk

# Garante que as stopwords estão disponíveis
nltk.download("stopwords")
stopwords_pt = stopwords.words("portuguese")

# Carrega o dataset
df = pd.read_csv("pre-processed.csv")

# Cria pipeline com stopwords em português
modelo = Pipeline([
    ("vetorizador", TfidfVectorizer(stop_words=stopwords_pt, max_features=5000)),
    ("classificador", MultinomialNB())
])

# Treina o modelo
modelo.fit(df["preprocessed_news"], df["label"])

# Salva
joblib.dump(modelo, "modelo.pkl")
print("Modelo treinado com sucesso!")
