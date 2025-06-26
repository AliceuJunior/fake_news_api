import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Carrega o dataset novo
df = pd.read_csv("fake_br.csv")

# Cria o pipeline: vetorizador + classificador
modelo = Pipeline([
    ("vetorizador", TfidfVectorizer(stop_words="portuguese", max_features=5000)),
    ("classificador", MultinomialNB())
])

# Treina o modelo
modelo.fit(df["texto"], df["rotulo"])

# Salva o modelo treinado
joblib.dump(modelo, "modelo.pkl")
print("Modelo treinado com Fake.Br salvo com sucesso!")
