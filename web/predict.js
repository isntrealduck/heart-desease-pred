const form = document.querySelector("#prediction-form");
const fields = document.querySelector("#feature-fields");
const result = document.querySelector("#prediction-result");
const submitButton = document.querySelector("#submit-button");

function makeField(field) {
  const { name, label: fieldLabel, kind, options, description, unit } = field;
  const label = document.createElement("label");
  label.className = "feature-field";
  const heading = document.createElement("span");
  heading.className = "feature-label-line";
  const title = document.createElement("span");
  title.textContent = fieldLabel;
  heading.append(title);
  if (unit) {
    const unitLabel = document.createElement("span");
    unitLabel.className = "feature-unit";
    unitLabel.textContent = unit;
    heading.append(unitLabel);
  }
  const info = document.createElement("span");
  info.className = "feature-info";
  const infoButton = document.createElement("button");
  infoButton.type = "button";
  infoButton.className = "info-trigger";
  infoButton.textContent = "i";
  infoButton.setAttribute("aria-label", `Information about ${fieldLabel}`);
  const tooltip = document.createElement("span");
  tooltip.className = "info-tooltip";
  tooltip.id = `tip-${schemaSafeId(name)}`;
  tooltip.setAttribute("role", "tooltip");
  tooltip.textContent = description;
  infoButton.setAttribute("aria-describedby", tooltip.id);
  info.append(infoButton, tooltip);
  heading.append(info);
  label.append(heading);

  let input;
  if (kind === "categorical") {
    input = document.createElement("select");
    input.name = name;
    input.append(new Option("Blank — use fitted imputation", ""));
    for (const option of options) input.append(new Option(option.label, option.value));
  } else {
    input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.inputMode = "decimal";
    input.name = name;
    input.placeholder = "Optional";
  }
  label.append(input);
  return label;
}

function schemaSafeId(value) {
  return value.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

async function loadSchema() {
  const response = await fetch("/api/schema");
  if (!response.ok) {
    throw new Error(
      `Prediction API unavailable (HTTP ${response.status}). Start the app with "python -m heart_ml.predict_web".`,
    );
  }
  const schema = await response.json();
  for (const name of schema.features) {
    fields.append(makeField({
      name,
      label: schema.feature_info[name].label,
      kind: schema.categorical.includes(name) ? "categorical" : "numeric",
      options: schema.category_options[name] || [],
      description: schema.feature_info[name].description,
      unit: schema.feature_info[name].unit,
    }));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.hidden = true;
  result.className = "prediction-result";
  submitButton.disabled = true;
  submitButton.textContent = "Calculating…";
  const payload = Object.fromEntries(
    [...form.elements]
      .filter((element) => element.name)
      .map((element) => [element.name, element.value === "" ? null : element.value]),
  );
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "Prediction failed.");
    result.classList.add(body.predicted_label === "Presence" ? "presence" : "absence");
    result.replaceChildren();
    const heading = document.createElement("strong");
    heading.textContent = `Model output: ${body.predicted_label}`;
    const score = document.createElement("span");
    score.textContent = `Model probability for Presence: ${(body.probability_presence * 100).toFixed(1)}%`;
    result.append(heading, score);
  } catch (error) {
    result.classList.add("error");
    result.textContent = error.message;
  } finally {
    result.hidden = false;
    submitButton.disabled = false;
    submitButton.textContent = "Get model output";
  }
});

document.querySelector("#clear-button").addEventListener("click", () => {
  result.hidden = true;
});

loadSchema().catch((error) => {
  fields.textContent = `${error.message} This page must be served by the prediction app, not a static file server.`;
  submitButton.disabled = true;
});
