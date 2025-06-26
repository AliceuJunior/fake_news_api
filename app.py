from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
modelo = joblib.load("modelo.pkl")

@app.route("/api/classificar-noticia", methods=["POST"])
def classificar_noticia():
    dados = request.get_json()
    texto = dados["texto"]
    previsao = modelo.predict([texto])[0]
    prob = modelo.predict_proba([texto]).max()

    return jsonify({
        "classe": previsao,
        "probabilidade": round(float(prob), 2)
    })

@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"status": "modelo carregado", "versao": "1.0"})

if __name__ == "__main__":
    app.run(debug=True)
