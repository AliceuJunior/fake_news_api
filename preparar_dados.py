import os
import pandas as pd

def ler_arquivos_pasta(pasta, rotulo):
    textos = []
    arquivos = os.listdir(pasta)
    
    for arquivo in arquivos:
        if arquivo.endswith(".txt"):
            caminho = os.path.join(pasta, arquivo)
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = f.read()
                textos.append((conteudo, rotulo))
    return textos

# Caminhos para as pastas
pasta_fake = "Fake.br-Corpus/fake"
pasta_true = "Fake.br-Corpus/true"

# Lê os arquivos
dados_falsos = ler_arquivos_pasta(pasta_fake, "falsa")
dados_verdadeiros = ler_arquivos_pasta(pasta_true, "verdadeira")

# Junta tudo e transforma em DataFrame
todos_os_dados = dados_falsos + dados_verdadeiros
df = pd.DataFrame(todos_os_dados, columns=["texto", "rotulo"])

# Embaralha os dados
df = df.sample(frac=1).reset_index(drop=True)

# Salva o CSV
df.to_csv("fake_br.csv", index=False)
print("Arquivo fake_br.csv criado com sucesso com", len(df), "notícias.")
