const apiURL = "http://localhost:8000/api";

document.getElementById("formulario").addEventListener("submit", async (e) => {
  e.preventDefault();

  const texto = document.getElementById("inputTexto").value;
  const resultadoDiv = document.getElementById("resultado");
  resultadoDiv.classList.add("oculto");
  resultadoDiv.innerHTML = "Verificando...";

  try {
    const res = await fetch(`${apiURL}/classificar-noticia`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto }),
    });

    const dados = await res.json();

    resultadoDiv.innerHTML = `
      <h3>Resultado</h3>
      <p><strong>Classe:</strong> ${dados.classe}</p>
      <p><strong>Probabilidade:</strong> ${dados.probabilidade}</p>
      <p><strong>Palavras influentes:</strong> ${dados.palavras_influentes.join(
        ", "
      )}</p>
    `;
    resultadoDiv.classList.remove("oculto");

    carregarHistorico();
  } catch (error) {
    resultadoDiv.innerHTML = "Erro ao verificar a notícia.";
  }
});

async function carregarHistorico() {
  const historicoDiv = document.getElementById("historico");

  try {
    const res = await fetch(`${apiURL}/historico`);
    const historico = await res.json();

    historicoDiv.innerHTML = historico
      .slice()
      .reverse() 
      .map(
        (item) => `
    <div class="historico-item">
      <p><strong>Texto:</strong> ${item.texto}</p>
      <p><strong>Classe:</strong> ${item.classe}</p>
      <p><strong>Probabilidade:</strong> ${item.probabilidade}</p>
    </div>
  `
      )
      .join("");
  } catch (error) {
    historicoDiv.innerHTML = "Erro ao carregar histórico.";
  }
}

window.onload = carregarHistorico;
