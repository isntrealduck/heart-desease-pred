const percentMetrics = new Set(["accuracy", "precision", "recall"]);
const displayMetrics = [
  ["roc_auc", "ROC-AUC"],
  ["average_precision", "Avg. precision"],
  ["accuracy", "Accuracy"],
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1"],
];

function formatScore(value, key) {
  return percentMetrics.has(key)
    ? `${(value * 100).toFixed(2)}%`
    : value.toFixed(4);
}

async function loadResults() {
  const response = await fetch("./results.json");
  if (!response.ok) throw new Error(`Could not load results (${response.status})`);
  return response.json();
}

function renderComparison(data) {
  const body = document.querySelector("#comparison-body");
  const bestByMetric = Object.fromEntries(displayMetrics.map(([key]) => [
    key,
    Math.max(...data.cv.map((row) => row.metrics[key].mean)),
  ]));
  const winner = [...data.cv].sort(
    (a, b) => b.metrics.roc_auc.mean - a.metrics.roc_auc.mean,
  )[0];

  document.querySelector("#hero-score").textContent =
    winner.metrics.roc_auc.mean.toFixed(4);
  document.querySelector("#hero-model").textContent =
    `${winner.label} · highest mean CV ROC-AUC`;

  for (const model of data.cv) {
    const row = document.createElement("tr");
    if (model.id === winner.id) row.className = "is-winner";
    const name = document.createElement("td");
    name.textContent = model.label;
    if (model.id === winner.id) {
      const badge = document.createElement("span");
      badge.className = "winner-pill";
      badge.textContent = "TOP CV AUC";
      name.append(" ", badge);
    }
    row.append(name);
    for (const [key] of displayMetrics) {
      const cell = document.createElement("td");
      const mean = model.metrics[key].mean;
      const average = document.createElement("span");
      average.className = "metric-value";
      average.textContent = formatScore(mean, key);
      const deviation = document.createElement("small");
      deviation.className = "metric-sd";
      deviation.textContent = `± ${model.metrics[key].fold_sd.toFixed(6)}`;
      cell.append(average, deviation);
      if (mean === bestByMetric[key]) cell.classList.add("best");
      row.append(cell);
    }
    body.append(row);
  }
}

function renderHoldout(data) {
  const { metrics } = data.holdout;
  document.querySelector("#holdout-auc").textContent =
    metrics.roc_auc.toFixed(4);
  const grid = document.querySelector("#holdout-grid");
  for (const [key, label] of displayMetrics.filter(([key]) => key !== "roc_auc")) {
    const item = document.createElement("div");
    item.className = "holdout-metric";
    const title = document.createElement("span");
    title.textContent = label.toUpperCase();
    const value = document.createElement("strong");
    value.textContent = formatScore(metrics[key], key);
    item.append(title, value);
    grid.append(item);
  }

  const matrix = document.querySelector("#confusion-matrix");
  const values = [
    ["", "PREDICTED ABSENCE", "PREDICTED PRESENCE"],
    ["ACTUAL ABSENCE", { count: metrics.confusion_matrix.tn, code: "TN", name: "True negative" },
      { count: metrics.confusion_matrix.fp, code: "FP", name: "False positive" }],
    ["ACTUAL PRESENCE", { count: metrics.confusion_matrix.fn, code: "FN", name: "False negative" },
      { count: metrics.confusion_matrix.tp, code: "TP", name: "True positive" }],
  ];
  values.forEach((row, rowIndex) => row.forEach((value, colIndex) => {
    const cell = document.createElement("div");
    if (typeof value === "object") {
      const count = document.createElement("strong");
      count.textContent = value.count.toLocaleString("en-US");
      const meaning = document.createElement("small");
      meaning.textContent = `${value.code} · ${value.name}`;
      cell.append(count, meaning);
    } else {
      cell.textContent = value;
    }
    if (rowIndex === 0 || colIndex === 0) cell.className = "matrix-head";
    if (rowIndex > 0 && colIndex === 0) cell.className = "matrix-label";
    if ((rowIndex === 1 && colIndex === 1) || (rowIndex === 2 && colIndex === 2)) {
      cell.classList.add("true-cell");
    }
    matrix.append(cell);
  }));
}

loadResults()
  .then((data) => {
    renderComparison(data);
    renderHoldout(data);
  })
  .catch((error) => {
    document.querySelector("#hero-model").textContent = "Results unavailable";
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.textContent = `${error.message}. Serve this folder over HTTP.`;
    row.append(cell);
    document.querySelector("#comparison-body").replaceChildren(row);
  });
