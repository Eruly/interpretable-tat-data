/* ================================================================
   Qwen3-VL TAT-QA – Frontend Logic
   ================================================================ */

// DOM refs
const sampleSelector = document.getElementById("sample-selector");
const btnLoad = document.getElementById("btn-load");
const btnInfer = document.getElementById("btn-infer");
const questionInput = document.getElementById("question-input");
const paragraphsDisplay = document.getElementById("paragraphs-display");
const spinner = document.getElementById("spinner");
const tableImg = document.getElementById("table-img");
const stage1Reasoning = document.getElementById("stage1-reasoning");
const finalAnswer = document.getElementById("final-answer");
const stage2Ocr = document.getElementById("stage2-ocr");
const stage3Raw = document.getElementById("stage3-raw");
const groundingTableImg = document.getElementById("grounding-table-img");
const groundingStage3 = document.getElementById("grounding-stage3-text");
const groundingOutputText = document.getElementById("grounding-output-text");
const bboxOverlay = document.getElementById("bbox-overlay");

let currentImageUrl = "";
let lastInferResult = null;

// ---- Tab switching ----
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// ---- Load samples list ----
async function loadSampleList() {
  try {
    const res = await fetch("/api/samples");
    const items = await res.json();
    sampleSelector.innerHTML = items
      .map((s) => `<option value="${s.id}">[${s.id}] ${escapeHtml(s.question.slice(0, 60))}</option>`)
      .join("");
  } catch (e) {
    sampleSelector.innerHTML = `<option>Error loading samples</option>`;
  }
}
loadSampleList();

// ---- Load & Render Table ----
btnLoad.addEventListener("click", async () => {
  const idx = sampleSelector.value;
  if (idx === "") return;
  btnLoad.disabled = true;
  btnLoad.textContent = "Loading...";
  try {
    const res = await fetch(`/api/load_sample/${idx}`);
    const data = await res.json();
    currentImageUrl = data.image_url;
    tableImg.src = data.image_url;
    groundingTableImg.src = data.image_url;
    questionInput.value = data.question;
    paragraphsDisplay.textContent = data.paragraphs || "(no paragraphs)";
    clearResults();
  } catch (e) {
    alert("Failed to load sample: " + e.message);
  } finally {
    btnLoad.disabled = false;
    btnLoad.textContent = "Load & Render Table";
  }
});

// ---- Run Inference ----
btnInfer.addEventListener("click", async () => {
  if (!currentImageUrl) { alert("Please load a sample first."); return; }
  const question = questionInput.value.trim();
  if (!question) { alert("Please enter a question."); return; }

  btnInfer.disabled = true;
  spinner.classList.remove("hidden");
  clearResults();

  try {
    const res = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_url: currentImageUrl, question }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Inference failed");
    }
    const data = await res.json();
    lastInferResult = data;
    renderResults(data);
  } catch (e) {
    finalAnswer.textContent = "Error: " + e.message;
  } finally {
    btnInfer.disabled = false;
    spinner.classList.add("hidden");
  }
});

// ---- Render Inference Results ----
function renderResults(data) {
  stage1Reasoning.textContent = data.stage1_reasoning || "(none)";
  finalAnswer.textContent = data.final_answer || "(none)";

  if (data.ocr_error) {
    stage2Ocr.textContent = "OCR Error: " + data.ocr_error;
  } else if (data.ocr_raw) {
    stage2Ocr.textContent = data.ocr_raw;
  } else if (data.ocr_items && data.ocr_items.length) {
    stage2Ocr.textContent = JSON.stringify(data.ocr_items, null, 2);
  } else {
    stage2Ocr.textContent = "(no OCR result)";
  }

  stage3Raw.textContent = data.stage3_result || "(none)";
  renderGroundingView(data.stage3_result || "");
  groundingOutputText.textContent = data.final_answer || "(none)";
}

function clearResults() {
  stage1Reasoning.textContent = "";
  finalAnswer.textContent = "";
  stage2Ocr.textContent = "";
  stage3Raw.textContent = "";
  groundingStage3.innerHTML = "";
  groundingOutputText.textContent = "";
  hideBboxOverlay();
  lastInferResult = null;
}

// ================================================================
// Stage 3 Grounding View – bbox parsing & rendering
// ================================================================

const PALETTE = [
  [255, 0, 0], [0, 200, 0], [0, 0, 255],
  [255, 165, 0], [128, 0, 255], [0, 180, 180],
  [220, 20, 60], [34, 139, 34], [255, 105, 180],
];

function renderGroundingView(text) {
  if (!text) { groundingStage3.innerHTML = "<em>No Stage 3 text.</em>"; return; }

  const entries = parseBboxEntries(text);
  if (!entries.length) {
    groundingStage3.innerHTML = formatPlainText(text);
    return;
  }

  const coordMap = {};
  let colorIdx = 0;
  function colorFor(norm) {
    const key = norm.join(",");
    if (!(key in coordMap)) { coordMap[key] = colorIdx++ % PALETTE.length; }
    return PALETTE[coordMap[key]];
  }

  let html = "";
  let cursor = 0;
  entries.forEach((e) => {
    const [r, g, b] = colorFor(e.bbox_norm);
    const before = text.slice(cursor, e.start);
    html += formatPlainText(before);

    const displayText = e.boldLabel
      ? `<b>${escapeHtml(e.boldLabel)}</b>`
      : escapeHtml(e.label);

    html += `<span class="bbox-ref" `
      + `data-x1="${e.bbox_norm[0]}" data-y1="${e.bbox_norm[1]}" `
      + `data-x2="${e.bbox_norm[2]}" data-y2="${e.bbox_norm[3]}" `
      + `data-color="${r},${g},${b}" `
      + `style="background:rgba(${r},${g},${b},.18); border-bottom:2px solid rgb(${r},${g},${b}); `
      + `color:rgb(${Math.max(0,r-60)},${Math.max(0,g-60)},${Math.max(0,b-60)});">`
      + displayText + `</span>`;
    cursor = e.end;
  });
  html += formatPlainText(text.slice(cursor));
  groundingStage3.innerHTML = html;

  bindBboxHover();
}

function formatPlainText(s) {
  let escaped = escapeHtml(s);
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  return escaped.replace(/\n/g, "<br>");
}

function parseBboxEntries(text) {
  const re = /(\*\*([^*]+)\*\*)?\s*<bbox>\s*([\s\S]*?)\s*<\/bbox>/g;
  const coordsRe = /\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]/;
  const entries = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    const boldRaw = m[1] || "";
    const boldLabel = m[2] || "";
    const inner = m[3];
    const cr = coordsRe.exec(inner);
    if (!cr) continue;
    let coords = [parseFloat(cr[1]), parseFloat(cr[2]), parseFloat(cr[3]), parseFloat(cr[4])];
    if (Math.max(...coords) > 1.1) coords = coords.map((c) => c / 1000);
    entries.push({
      start: m.index,
      end: m.index + m[0].length,
      raw: m[0],
      label: inner,
      boldLabel: boldLabel,
      bbox_norm: coords,
    });
  }
  return entries;
}

// ---- Bbox hover overlay ----
function bindBboxHover() {
  document.querySelectorAll("#grounding-stage3-text .bbox-ref").forEach((span) => {
    span.addEventListener("mouseenter", () => showBboxOverlay(span));
    span.addEventListener("mouseleave", () => hideBboxOverlay());
  });
}

function getRenderedImageRect(imgEl) {
  const elemRect = imgEl.getBoundingClientRect();
  const natW = imgEl.naturalWidth;
  const natH = imgEl.naturalHeight;
  if (!natW || !natH) return elemRect;

  const elemAspect = elemRect.width / elemRect.height;
  const imgAspect = natW / natH;

  let renderW, renderH, offsetX, offsetY;
  if (imgAspect > elemAspect) {
    renderW = elemRect.width;
    renderH = elemRect.width / imgAspect;
    offsetX = 0;
    offsetY = (elemRect.height - renderH) / 2;
  } else {
    renderH = elemRect.height;
    renderW = elemRect.height * imgAspect;
    offsetX = (elemRect.width - renderW) / 2;
    offsetY = 0;
  }
  return {
    left: elemRect.left + offsetX,
    top: elemRect.top + offsetY,
    width: renderW,
    height: renderH,
  };
}

function showBboxOverlay(span) {
  const imgEl = groundingTableImg;
  if (!imgEl || !imgEl.naturalWidth) return;

  const x1 = parseFloat(span.dataset.x1);
  const y1 = parseFloat(span.dataset.y1);
  const x2 = parseFloat(span.dataset.x2);
  const y2 = parseFloat(span.dataset.y2);
  const color = span.dataset.color || "0,0,255";

  const normX1 = x1 > 1.1 ? x1 / 1000 : x1;
  const normY1 = y1 > 1.1 ? y1 / 1000 : y1;
  const normX2 = x2 > 1.1 ? x2 / 1000 : x2;
  const normY2 = y2 > 1.1 ? y2 / 1000 : y2;

  const ir = getRenderedImageRect(imgEl);
  const pxLeft = ir.left + normX1 * ir.width;
  const pxTop = ir.top + normY1 * ir.height;
  const pxW = (normX2 - normX1) * ir.width;
  const pxH = (normY2 - normY1) * ir.height;

  const [r, g, b] = color.split(",").map(Number);
  bboxOverlay.style.left = pxLeft + "px";
  bboxOverlay.style.top = pxTop + "px";
  bboxOverlay.style.width = pxW + "px";
  bboxOverlay.style.height = pxH + "px";
  bboxOverlay.style.borderColor = `rgb(${r},${g},${b})`;
  bboxOverlay.style.backgroundColor = `rgba(${r},${g},${b},0.22)`;
  bboxOverlay.style.display = "block";
}

function hideBboxOverlay() {
  bboxOverlay.style.display = "none";
}

// ---- Helpers ----
function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}
