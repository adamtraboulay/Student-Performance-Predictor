const form = document.getElementById("student-form");
const knowGrades = document.getElementById("know-grades");
const gradeInputs = document.getElementById("grade-inputs");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");

knowGrades.addEventListener("change", () => {
  gradeInputs.hidden = !knowGrades.checked;
  if (!knowGrades.checked) {
    form.G1.value = "";
    form.G2.value = "";
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  resultBox.hidden = true;

  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = value;
  }
  if (!knowGrades.checked) {
    delete payload.G1;
    delete payload.G2;
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    if (!response.ok) {
      showError(data.errors ? data.errors.join(", ") : "Something went wrong.");
      return;
    }

    showResult(data);
  } catch (err) {
    showError("Couldn't reach the server. Try again in a moment.");
  }
});

function showResult(data) {
  resultBox.querySelector(".result-grade").textContent = `Predicted grade: ${data.prediction} / 20`;

  const categoryEl = resultBox.querySelector(".result-category");
  categoryEl.textContent = `Category: ${data.category}`;
  categoryEl.className = `result-category ${data.category.toLowerCase()}`;

  const modelLabel = data.model === "full"
    ? "Predicted using the recent-grades model (knows G1/G2)"
    : "Predicted using the behavior-only model (no prior grades)";
  resultBox.querySelector(".result-model").textContent = modelLabel;

  resultBox.hidden = false;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
