/* SatQuery AI — framework-free ES module. All untrusted text uses textContent. */
const $ = (id) => document.getElementById(id);
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MAX_RUNS = 12;
const STORAGE = { key: "satquery.hfToken", model: "satquery.model", access: "satquery.access" };
const MODES = {
  auto: "The agent selects a compatible tool from your question and image types.",
  single: "Ask a question, describe a scene, or locate a region in image A.",
  bitemporal: "Compare the same area and sensor modality. A is earlier; B is later.",
  fusion: "Combine one optical or multispectral image with one corresponding SAR image.",
};
const QUESTIONS = {
  auto: ["Describe the dominant land cover.", "What evidence supports your interpretation?"],
  single: ["Describe the dominant land cover.", "Locate the built-up areas."],
  bitemporal: ["What changed between these acquisitions?", "Where are the strongest candidate changes?"],
  fusion: ["Compare optical patterns with SAR backscatter.", "Which observations agree across the two sensors?"],
};

function readStorage(storage, key) {
  try { return window[storage].getItem(key) || ""; } catch { return ""; }
}

function writeStorage(storage, key, value) {
  try {
    if (value) window[storage].setItem(key, value); else window[storage].removeItem(key);
    return true;
  } catch { return false; }
}

function freshSlot() {
  return { file: null, preview: null, metadata: null, warnings: [], status: "empty", controller: null, generation: 0, timer: null };
}

function sessionId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  // getRandomValues also works when a non-HTTPS page needs to show its error.
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

const state = {
  id: sessionId(), startedAt: new Date().toISOString(), mode: "auto", slots: [freshSlot(), freshSlot()],
  busy: false, connected: false, health: null, controller: null, runs: [], messages: [], selectedRun: null,
  selectedEvidence: null, dialogEvidence: null,
  token: readStorage("localStorage", STORAGE.key), model: readStorage("localStorage", STORAGE.model),
  access: readStorage("sessionStorage", STORAGE.access), remember: true,
};
let toastTimer;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = String(text);
  return element;
}

function toast(message, error = false) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 8000 : 5000);
}

function secureOrigin() {
  return location.protocol === "https:" || (location.protocol === "http:" && ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname));
}

async function api(path, { method = "GET", body, signal, token = state.token, access = state.access } = {}) {
  if (!secureOrigin()) throw new Error("Open this workspace over HTTPS, or run it on localhost. Opening index.html directly is unsupported.");
  const headers = { Accept: "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  if (access) headers["X-SatQuery-Access"] = access;
  const response = await fetch(path, { method, body, signal, headers, credentials: "same-origin", redirect: "error", cache: "no-store" });
  let data;
  try { data = await response.json(); } catch (error) { if (error.name === "AbortError") throw error; throw new Error(`The backend returned an unreadable response (HTTP ${response.status}).`); }
  if (!response.ok) {
    const error = new Error(data.error?.message || `The request failed (HTTP ${response.status}).`);
    error.code = data.error?.code || "request_failed";
    error.requestId = data.error?.request_id;
    throw error;
  }
  return data;
}

function activeIndices() {
  return state.mode === "single" ? [0] : (state.slots[1].file || ["bitemporal", "fusion"].includes(state.mode) ? [0, 1] : [0]);
}

function currentPairMode() {
  if (state.mode !== "auto") return state.mode;
  if (!state.slots[1].file) return "single";
  return [$("sensor-0").value, $("sensor-1").value].filter((v) => v === "sar").length === 1 ? "fusion" : "bitemporal";
}

function getSpec(index) {
  const modality = $(`sensor-${index}`).value;
  const rawBands = $(`bands-${index}`).value.trim();
  let bands = null;
  if (rawBands) {
    const parts = rawBands.split(",").map((v) => v.trim());
    if (parts.length !== 3 || parts.some((v) => !/^\d{1,2}$/.test(v) || Number(v) < 1 || Number(v) > 64)) {
      throw new Error(`Image ${index ? "B" : "A"}: display bands need three indices from 1 to 64, such as 3,2,1.`);
    }
    bands = parts.map(Number);
  }
  const optionalBand = (id) => {
    const text = $(id).value.trim();
    if (!text) return null;
    const value = Number(text);
    if (!Number.isInteger(value) || value < 1 || value > 64) throw new Error("Red and NIR indices must be whole numbers from 1 to 64.");
    return value;
  };
  const red = modality === "sar" ? null : optionalBand(`red-${index}`);
  const nir = modality === "sar" ? null : optionalBand(`nir-${index}`);
  if ((red === null) !== (nir === null)) throw new Error("Set both red and NIR band indices, or leave both empty.");
  if (red !== null && red === nir) throw new Error("Red and NIR must refer to different bands.");
  return { modality, bands, red_band: red, nir_band: nir,
    sar_scale: modality === "sar" ? $(`sar-scale-${index}`).value : "display",
    acquired_at: $(`date-${index}`).value || null };
}

function renderControls() {
  const indices = activeIndices();
  const used = indices.filter((i) => state.slots[i].file).length;
  const pairMode = currentPairMode();
  $("image-count").textContent = `${used} / ${state.mode === "single" ? 1 : 2}`;
  $("analysis-count").textContent = state.runs.length;
  $("mode").value = state.mode;
  $("mode-description").textContent = MODES[state.mode];
  document.querySelectorAll("[data-mode]").forEach((button) => {
    const selected = button.dataset.mode === state.mode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.disabled = state.busy;
  });
  const hasPair = indices.length === 2;
  const hasUnlocated = hasPair && indices.some((i) => !state.slots[i].metadata?.crs);
  $("alignment-control").hidden = !hasUnlocated;
  $("threshold-control").hidden = pairMode !== "bitemporal";
  document.querySelector(".sidebar").classList.toggle("mobile-controls", hasUnlocated || pairMode === "bitemporal");
  const titles = pairMode === "bitemporal" ? ["Earlier acquisition", "Later acquisition"] : pairMode === "fusion" ? ["Reference image", "Co-registered sensor"] : ["Primary image", "Comparison image"];
  titles.forEach((title, i) => { $(`slot-title-${i}`).textContent = i === 1 && state.mode === "single" ? "Inactive in single mode" : title; });
  const secondary = document.querySelector('[data-slot="1"]');
  secondary.classList.toggle("disabled-slot", state.mode === "single");
  $(`drop-1`).setAttribute("aria-disabled", String(state.mode === "single" || state.busy));
  $("drop-0").setAttribute("aria-disabled", String(state.busy));
  $("model-label").textContent = state.model ? state.model.split("/").slice(1).join("/") : "Not connected";
  $("key-status").textContent = state.token ? "Personal token configured" : state.health?.server_token_configured ? "Server token configured" : "Add a Hugging Face token";
  const waiting = indices.some((i) => state.slots[i].status === "loading");
  $("analyze-button").disabled = state.busy || waiting || !state.connected;
  $("analyze-button").hidden = state.busy;
  $("cancel-analysis").hidden = !state.busy;
  $("composer-hint").textContent = state.busy ? "Processing imagery and model response" : waiting ? "Waiting for raster validation" : "⌘ / Ctrl + Enter to analyze";
  $("copy-trace").disabled = !state.selectedRun;
  $("export-menu-button").disabled = state.busy || !state.messages.length;
  $("download-json").disabled = state.busy || !state.messages.length;
  $("print-pdf").disabled = state.busy || !state.messages.length;
  for (const id of ["mode", "threshold", "co-registered", "query", "new-session", "new-session-mobile", "open-settings"]) $(id).disabled = state.busy;
  for (let i = 0; i < 2; i++) {
    const inactive = i === 1 && state.mode === "single";
    for (const name of ["file", "sensor", "date", "bands", "red", "nir", "sar-scale"]) $(`${name}-${i}`).disabled = state.busy || inactive;
    document.querySelector(`[data-remove="${i}"]`).disabled = state.busy || !state.slots[i].file;
    document.querySelector(`[data-zoom="${i}"]`).disabled = !state.slots[i].preview;
  }
  document.querySelectorAll(".suggestion").forEach((b) => { b.disabled = state.busy; });
}

function renderSuggestions() {
  $("suggestions").replaceChildren();
  for (const question of QUESTIONS[currentPairMode()] || QUESTIONS.auto) {
    const button = node("button", "suggestion", `${question} ↗`);
    button.type = "button";
    button.addEventListener("click", () => { $("query").value = question; $("query").focus(); });
    $("suggestions").append(button);
  }
}

function setMode(mode) {
  if (state.busy || !MODES[mode]) return;
  state.mode = mode;
  $("co-registered").checked = false;
  if (mode === "fusion" && !state.slots[1].file && $("sensor-0").value !== "sar") {
    $("sensor-1").value = "sar";
    renderSensorFields(1);
  }
  renderSuggestions();
  renderControls();
}

function renderSensorFields(index) {
  const sensor = $(`sensor-${index}`).value;
  $(`sar-control-${index}`).hidden = sensor !== "sar";
  $(`spectral-${index}`).hidden = sensor === "sar";
  if (sensor !== "optical") $(`bands-details-${index}`).open = true;
}

function renderSlot(index) {
  const slot = state.slots[index];
  const preview = $(`preview-${index}`);
  preview.hidden = !slot.preview;
  if (slot.preview) preview.src = slot.preview; else preview.removeAttribute("src");
  $(`empty-${index}`).hidden = Boolean(slot.preview);
  $(`loading-${index}`).hidden = slot.status !== "loading";
  $(`drop-${index}`).classList.toggle("has-image", Boolean(slot.preview));
  $(`filename-${index}`).textContent = slot.file?.name || "No image selected";
  $(`filename-${index}`).title = slot.file?.name || "";
  const metadata = $(`metadata-${index}`);
  metadata.replaceChildren();
  if (slot.metadata) {
    const m = slot.metadata;
    const values = [`${m.width} × ${m.height}`, `${m.band_count} bands`, m.crs || "No CRS", `${(m.valid_pixel_fraction * 100).toFixed(1)}% valid`];
    values.forEach((text) => metadata.append(node("span", "", text)));
    metadata.title = slot.warnings.join("\n");
  } else metadata.append(node("span", "", slot.file ? "Validation required" : "Raster metadata appears after upload"));
  renderControls();
}

function setSlotError(index, message = "") {
  $(`error-${index}`).textContent = message;
  $(`error-${index}`).hidden = !message;
}

async function validatePreview(index) {
  const slot = state.slots[index];
  if (!slot.file) return;
  slot.controller?.abort();
  const generation = ++slot.generation;
  const controller = new AbortController();
  slot.controller = controller;
  slot.status = "loading";
  slot.metadata = null;
  setSlotError(index);
  renderSlot(index);
  const timer = setTimeout(() => controller.abort(), 45000);
  try {
    const spec = getSpec(index);
    const form = new FormData();
    form.append("images", slot.file, slot.file.name);
    form.append("payload", JSON.stringify(spec));
    const data = await api("/api/preview", { method: "POST", body: form, signal: controller.signal });
    if (state.slots[index] !== slot || generation !== slot.generation) return;
    slot.preview = data.preview;
    slot.metadata = data.metadata;
    slot.warnings = data.warnings || [];
    slot.status = "ready";
  } catch (error) {
    if (state.slots[index] !== slot || generation !== slot.generation) return;
    slot.status = "error";
    slot.preview = null;
    setSlotError(index, error.name === "AbortError" ? "Preview validation timed out. Try a smaller raster or select the file again." : error.message);
    if (error.code === "workspace_auth") openSettings();
    if ($(`sensor-${index}`).value === "multispectral") $(`bands-details-${index}`).open = true;
  } finally {
    clearTimeout(timer);
    if (state.slots[index] === slot && generation === slot.generation) {
      slot.controller = null;
      renderSlot(index);
      renderSuggestions();
    }
  }
}

function queuePreview(index) {
  if (!state.slots[index].file || state.busy) return;
  const slot = state.slots[index];
  clearTimeout(slot.timer);
  slot.controller?.abort();
  slot.generation++;
  slot.status = "loading";
  slot.metadata = null;
  $("co-registered").checked = false;
  renderSlot(index);
  slot.timer = setTimeout(() => validatePreview(index), 350);
}

function removeImage(index) {
  const previous = state.slots[index];
  previous.controller?.abort();
  clearTimeout(previous.timer);
  state.slots[index] = freshSlot();
  $(`file-${index}`).value = "";
  $(`date-${index}`).value = "";
  $("co-registered").checked = false;
  setSlotError(index);
  renderSlot(index);
  renderSuggestions();
}

async function acceptFiles(files, index) {
  if (state.busy) return;
  const incoming = Array.from(files);
  if (!incoming.length) return;
  if (incoming.length > 2) return toast("Choose at most two images per analysis.", true);
  for (const file of incoming) {
    if (!/\.(tif|tiff|png|jpe?g)$/i.test(file.name)) return toast("Choose a GeoTIFF, TIFF, PNG or JPEG file.", true);
    if (!file.size || file.size > MAX_FILE_BYTES) return toast(`${file.name}: each file must be nonempty and at most 20 MiB.`, true);
  }
  if (incoming.length === 2 && state.mode === "single") setMode("auto");
  if (incoming.length === 1 && index === 1 && state.mode === "single") return;
  const targets = incoming.length === 2 ? [0, 1] : [index];
  for (let j = 0; j < incoming.length; j++) {
    const i = targets[j];
    const previous = state.slots[i];
    previous.controller?.abort();
    clearTimeout(previous.timer);
    if (previous.file) $(`date-${i}`).value = "";
    state.slots[i] = { ...freshSlot(), file: incoming[j] };
    setSlotError(i);
    renderSlot(i);
  }
  $("co-registered").checked = false;
  await Promise.allSettled(targets.map((i) => validatePreview(i)));
}

function message(role, text, extra = {}) {
  const item = { role, content: text, created_at: new Date().toISOString(), ...extra };
  state.messages.push(item);
  $("chat-welcome").hidden = true;
  const article = node("article", `chat-message ${role}`);
  const header = node("div", "message-header");
  header.append(node("span", "message-avatar", role === "user" ? "A" : "✳"));
  header.append(node("strong", "", role === "user" ? "You" : "SatQuery AI"));
  const time = node("time", "", new Date(item.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
  time.dateTime = item.created_at;
  header.append(time);
  article.append(header, node("p", role === "error" ? "error-message" : "message-text", text));
  $("chat-log").append(article);
  scrollChat();
  return article;
}

function scrollChat() {
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}

function renderAnswer(run) {
  const result = run.result;
  const article = message("assistant", result.answer, { request_id: result.request_id, scene_key: run.sceneKey });
  const meta = node("div", "response-meta");
  meta.append(node("span", "", result.task.replaceAll("_", " ")));
  meta.append(node("span", "confidence-chip", `${Math.round(result.confidence_score * 100)}% support · heuristic`));
  article.append(meta);
  if (result.observations.length) {
    const observations = node("ul", "response-observations");
    for (const item of result.observations) {
      const li = node("li", "", item.description);
      li.append(node("span", "observation-citation", `[${item.evidence_ids.join(", ")}]`));
      observations.append(li);
    }
    article.append(observations);
  }
  if (result.warnings?.length) {
    const details = node("details", "message-warnings");
    details.append(node("summary", "", `${result.warnings.length} limitations & assumptions`));
    const ul = node("ul");
    result.warnings.forEach((warning) => ul.append(node("li", "", warning)));
    details.append(ul);
    article.append(details);
  }
  const inspect = node("button", "inspect-button", "Inspect this analysis ↗");
  inspect.type = "button";
  inspect.addEventListener("click", () => { selectRun(run); $("evidence-title").scrollIntoView({ behavior: "smooth", block: "start" }); });
  article.append(inspect);
  scrollChat();
}

function pendingMessage() {
  const container = node("div", "pending-message");
  container.id = "pending-message";
  container.append(node("span", "spinner"), node("span", "", "Calculating evidence and waiting for the vision model…"));
  $("chat-log").append(container);
  scrollChat();
}

function sceneKey(indices, specs) {
  return JSON.stringify({ mode: currentPairMode(), images: indices.map((i, j) => ({ sha256: state.slots[i].metadata.sha256, spec: specs[j] })) });
}

function conversationContext(key) {
  const history = state.runs.filter((run) => run.sceneKey === key).flatMap((run) => [
    { role: "user", content: run.query.slice(0, 4000) }, { role: "assistant", content: run.result.answer.slice(0, 4000) },
  ]).slice(-8);
  while (history.reduce((n, m) => n + m.content.length, 0) > 12000) history.shift();
  return history;
}

async function analyze(event) {
  event.preventDefault();
  if (state.busy) return;
  const query = $("query").value.trim();
  if (query.length < 3) return toast("Enter a question with at least three characters.", true);
  if (state.runs.length >= MAX_RUNS) return toast(`This session has reached ${MAX_RUNS} analyses. Export it, then start a new session.`, true);
  if (!state.connected) return toast("The backend is unavailable. Start FastAPI and reload this page.", true);
  if (!state.token && !state.health?.server_token_configured) { openSettings(); return; }
  const indices = activeIndices();
  if (indices.some((i) => !state.slots[i].file)) return toast("Upload image A and any required comparison image before analyzing.", true);
  if (indices.some((i) => state.slots[i].status !== "ready")) return toast("Resolve the image validation messages before analyzing.", true);
  let specs;
  try { specs = indices.map(getSpec); } catch (error) { return toast(error.message, true); }
  if (indices.length === 2 && indices.some((i) => !state.slots[i].metadata.crs) && !$("co-registered").checked) {
    $("co-registered").focus();
    return toast("For images without a CRS, confirm that their pixel grids already cover the same area and are aligned.", true);
  }
  const key = sceneKey(indices, specs);
  const payload = { query, mode: state.mode, model: state.model || null, images: specs,
    co_registered: $("co-registered").checked, change_threshold: Number($("threshold").value), history: conversationContext(key) };
  const form = new FormData();
  indices.forEach((i) => form.append("images", state.slots[i].file, state.slots[i].file.name));
  form.append("payload", JSON.stringify(payload));
  message("user", query, { scene_key: key });
  state.busy = true;
  $("app-shell").classList.add("busy");
  renderControls();
  pendingMessage();
  const controller = new AbortController();
  state.controller = controller;
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 195000);
  try {
    const result = await api("/api/analyze", { method: "POST", body: form, signal: controller.signal });
    const run = { query, mode: state.mode, sceneKey: key, result, created_at: new Date().toISOString() };
    state.runs.push(run);
    $("pending-message")?.remove();
    renderAnswer(run);
    selectRun(run);
    $("query").value = "";
  } catch (error) {
    $("pending-message")?.remove();
    let text = error.message;
    if (error.name === "AbortError") text = timedOut ? "The browser reached its time limit. The provider may still be processing the request." : "Cancelled in this browser. A request already accepted by the provider may continue and use credits.";
    message("error", text, { scene_key: key, request_id: error.requestId || null });
    toast(text, true);
    if (["provider_auth", "missing_token", "workspace_auth"].includes(error.code)) openSettings();
  } finally {
    clearTimeout(timeout);
    state.busy = false;
    state.controller = null;
    $("app-shell").classList.remove("busy");
    renderControls();
    $("query").focus();
  }
}

function metric(label, value, detail = "") {
  const card = node("div", "metric-card");
  card.append(node("span", "", label), node("strong", "", value));
  if (detail) card.append(node("small", "", detail));
  return card;
}

function selectEvidence(item) {
  state.selectedEvidence = item;
  $("evidence-image").src = item.data_url;
  $("evidence-image").alt = item.label;
  $("evidence-caption").textContent = item.description;
  document.querySelectorAll(".evidence-tab").forEach((button) => {
    const selected = button.dataset.evidence === item.id;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function selectRun(run) {
  state.selectedRun = run;
  const result = run.result;
  $("evidence-empty").hidden = true;
  $("evidence-content").hidden = false;
  $("evidence-count").textContent = `${result.visual_evidence.length} evidence images`;
  $("evidence-tabs").replaceChildren();
  result.visual_evidence.forEach((item) => {
    const button = node("button", "evidence-tab", item.label);
    button.type = "button";
    button.dataset.evidence = item.id;
    button.addEventListener("click", () => selectEvidence(item));
    $("evidence-tabs").append(button);
  });
  const preferred = result.visual_evidence.find((e) => e.kind === "model_grounding") || result.visual_evidence.find((e) => ["change_overlay", "fusion_overlay"].includes(e.kind)) || result.visual_evidence[0];
  if (preferred) selectEvidence(preferred);
  const metrics = $("metrics-grid");
  metrics.replaceChildren(metric("Evidence confidence", `${Math.round(result.confidence_score * 100)}%`, "Heuristic · not calibrated accuracy"));
  const m = result.metrics;
  if (m.candidate_change_fraction !== undefined) metrics.append(metric("Candidate radiometric change", `${(m.candidate_change_fraction * 100).toFixed(2)}%`, `${m.compared_pixels.toLocaleString()} valid analysis pixels compared`));
  if (m.candidate_map_plane_area_m2 !== undefined && m.candidate_map_plane_area_m2 !== null) metrics.append(metric("Candidate map-plane area", `${(m.candidate_map_plane_area_m2 / 10000).toFixed(2)} ha`, "Approximate · projection distortion uncorrected"));
  if (m.joint_valid_fraction !== undefined) metrics.append(metric("Jointly valid pixels", `${(m.joint_valid_fraction * 100).toFixed(1)}%`, "On the reference analysis grid"));
  if (m.display_intensity_correlation !== undefined) metrics.append(metric("Display intensity correlation", m.display_intensity_correlation === null ? "Undefined" : m.display_intensity_correlation.toFixed(3), "Diagnostic only · not alignment accuracy"));
  for (const [key, value] of Object.entries(m)) {
    if (key.startsWith("ndvi_image_") && value.mean !== null) metrics.append(metric(`Mean NDVI · ${key.slice(-1).toUpperCase()}`, value.mean.toFixed(3), "From the declared red/NIR bands"));
  }
  $("region-list").replaceChildren();
  const counters = {};
  for (const region of result.regions) {
    counters[region.image_id] = (counters[region.image_id] || 0) + 1;
    const item = node("div", "region-entry");
    item.append(node("strong", "", `R${counters[region.image_id]} · ${region.image_id.slice(-1).toUpperCase()} · ${region.label}`));
    item.append(node("div", "", region.evidence));
    item.append(node("small", "", "Unverified model proposal"));
    $("region-list").append(item);
  }
  const trace = result.execution_trace;
  const overview = $("trace-overview");
  overview.replaceChildren();
  const entries = [["TOOL SELECTED", trace.tool], ["DURATION", `${(trace.duration_ms / 1000).toFixed(1)} s`], ["CONFIDENCE", `${Math.round(result.confidence_score * 100)}% · heuristic`]];
  entries.forEach(([label, value]) => {
    const item = node("div", `trace-stat${label === "CONFIDENCE" ? " confidence" : ""}`);
    item.append(node("small", "", label), node("strong", "", value));
    overview.append(item);
  });
  $("trace-details").hidden = false;
  $("trace-json").textContent = JSON.stringify(trace, null, 2);
  renderControls();
}

function showImage(item) {
  if (!item?.data_url) return;
  state.dialogEvidence = item;
  $("image-dialog-title").textContent = item.label;
  $("dialog-image").src = item.data_url;
  $("dialog-image").alt = item.label;
  $("dialog-image-description").textContent = item.description || "";
  $("image-dialog").showModal();
}

function openSettings() {
  $("api-key").value = state.token;
  $("model-input").value = state.model || state.health?.default_model || "Qwen/Qwen3-VL-30B-A3B-Instruct";
  $("remember-key").checked = state.remember;
  $("workspace-access").value = state.access;
  $("workspace-access-field").hidden = !state.health?.access_token_required;
  $("settings-status").textContent = "";
  $("settings-status").classList.remove("error");
  if (!$("settings-dialog").open) $("settings-dialog").showModal();
  if (state.health?.access_token_required && !state.access) $("workspace-access").focus(); else $("api-key").focus();
}

function settingsStatus(message, error = false) {
  $("settings-status").textContent = message;
  $("settings-status").classList.toggle("error", error);
}

function saveSettings(event) {
  event.preventDefault();
  const token = $("api-key").value.trim();
  const model = $("model-input").value.trim();
  const access = $("workspace-access").value.trim();
  if (token && !/^[A-Za-z0-9_.-]{1,256}$/.test(token)) return settingsStatus("The token has unsupported characters. Paste your Hugging Face token without a Bearer prefix.", true);
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*(?::[A-Za-z0-9_-]+)?$/.test(model)) return settingsStatus("Enter an organization/model ID, optionally followed by :provider.", true);
  if (!secureOrigin()) return settingsStatus("Tokens can be used only over HTTPS or localhost HTTP.", true);
  state.token = token;
  state.model = model;
  state.access = access;
  state.remember = $("remember-key").checked;
  const keySaved = writeStorage("localStorage", STORAGE.key, state.remember ? token : "");
  const modelSaved = writeStorage("localStorage", STORAGE.model, model);
  const accessSaved = writeStorage("sessionStorage", STORAGE.access, access);
  $("settings-dialog").close();
  toast(!keySaved && !state.remember ? "Settings apply to this tab, but the previous saved token could not be removed. Clear this site's browser storage." : keySaved && modelSaved && accessSaved ? "Settings saved. Your token is excluded from reports." : "Settings apply to this tab. Browser storage is unavailable, so some settings could not be remembered.");
  renderControls();
  for (let i = 0; i < 2; i++) if (state.slots[i].file && state.slots[i].status === "error") queuePreview(i);
}

async function refreshModels() {
  const button = $("refresh-models");
  button.disabled = true;
  button.textContent = "Loading models…";
  settingsStatus("Checking the Hugging Face vision-model catalog…");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 35000);
  try {
    const data = await api("/api/models", { token: $("api-key").value.trim(), access: $("workspace-access").value.trim(), signal: controller.signal });
    $("model-options").replaceChildren();
    for (const model of data.models) {
      const option = node("option");
      option.value = model.id;
      option.label = model.providers?.length ? model.providers.join(", ") : model.id;
      $("model-options").append(option);
    }
    settingsStatus(data.models.length ? `${data.models.length} image-capable models found. Choose one using the model field. The catalog does not guarantee multi-image support.` : "No image-capable models were returned. Check your account or enter a supported model ID.");
  } catch (error) {
    settingsStatus(error.name === "AbortError" ? "The model catalog request timed out." : error.message, true);
  } finally {
    clearTimeout(timer);
    button.disabled = false;
    button.textContent = "Refresh available models";
  }
}

function sessionReport() {
  // Explicit allowlist: no settings object, credentials, or original File objects.
  return {
    schema_version: "1.0", application: "SatQuery AI", version: state.health?.version || "1.0.0",
    session_id: state.id, started_at: state.startedAt, exported_at: new Date().toISOString(),
    notes: ["Confidence scores are uncalibrated evidence-support heuristics.", "Images are analysis-resolution visualizations; original source rasters are not embedded.", "Source and evidence SHA-256 hashes support byte-level provenance, not scientific validity."],
    conversation: state.messages.map(({ role, content, created_at, request_id }) => ({ role, content, created_at, ...(request_id ? { request_id } : {}) })),
    analyses: state.runs.map(({ query, mode, result, created_at }) => ({ query, mode, created_at, result })),
  };
}

function reportFilename(extension) {
  return `satquery-report-${new Date().toISOString().replaceAll(":", "-").slice(0, 19)}.${extension}`;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = node("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadJSON() {
  downloadBlob(new Blob([JSON.stringify(sessionReport(), null, 2)], { type: "application/json;charset=utf-8" }), reportFilename("json"));
  $("export-dialog").close();
  toast("The JSON report includes this session's chat, imagery evidence and execution traces.");
}

async function printPDF() {
  const report = sessionReport();
  const root = $("print-report");
  root.replaceChildren();
  root.append(node("h1", "", "SatQuery AI"), node("p", "report-meta", `Earth observation report · ${new Date(report.exported_at).toLocaleString()}`));
  root.append(node("p", "report-meta", `Session ${report.session_id} · ${report.analyses.length} completed analyses`));
  root.append(node("p", "", report.notes.join(" ")));
  root.append(node("h2", "", "Conversation"));
  for (const item of report.conversation) {
    root.append(node("h3", "", `${item.role === "user" ? "You" : item.role === "error" ? "Request error" : "SatQuery AI"} · ${new Date(item.created_at).toLocaleTimeString()}`));
    root.append(node("p", item.role === "error" ? "report-answer report-error" : "report-answer", item.content));
  }
  report.analyses.forEach((run, index) => {
    const result = run.result;
    const section = node("section", "report-run");
    section.append(node("h2", "", `Analysis ${index + 1} · ${result.task.replaceAll("_", " ")}`));
    section.append(node("p", "", run.query));
    section.append(node("p", "report-meta", `${result.tool} · ${result.execution_trace.model} · ${Math.round(result.confidence_score * 100)}% heuristic evidence support`));
    section.append(node("h3", "", "Observations"));
    for (const observation of result.observations) section.append(node("p", "report-observation", `${observation.description} [${observation.evidence_ids.join(", ")}]`));
    section.append(node("h3", "", "Visual evidence"));
    const images = node("div", "report-images");
    for (const evidence of result.visual_evidence) {
      const figure = node("figure");
      const image = node("img");
      image.src = evidence.data_url;
      image.alt = evidence.label;
      figure.append(image, node("figcaption", "", `${evidence.id} — ${evidence.description}`));
      images.append(figure);
    }
    section.append(images);
    if (result.regions.length) {
      section.append(node("h3", "", "Proposed regions"), node("pre", "", JSON.stringify(result.regions, null, 2)));
    }
    section.append(node("h3", "", "Measurements"), node("pre", "", JSON.stringify(result.metrics, null, 2)));
    section.append(node("h3", "", "Limitations and assumptions"));
    const limitations = node("ul", "report-limitations");
    result.warnings.forEach((text) => limitations.append(node("li", "", text)));
    section.append(limitations, node("h3", "", "Execution trace"), node("pre", "", JSON.stringify(result.execution_trace, null, 2)));
    root.append(section);
  });
  root.append(node("footer", "", "Generated from this SatQuery AI session. VLM interpretations and proposed boxes require independent validation."));
  $("export-dialog").close();
  // Decode real evidence before opening print preview; no remote images or scripts.
  const decoded = await Promise.allSettled(Array.from(root.querySelectorAll("img"), (img) => img.decode()));
  if (decoded.some((item) => item.status === "rejected")) return toast("One report image could not be decoded. Export JSON to preserve all evidence bytes.", true);
  const title = document.title;
  document.title = reportFilename("pdf").replace(/\.pdf$/, "");
  window.print();
  document.title = title;
}

function clearSession() {
  if (state.busy) return;
  if (state.messages.length && !window.confirm("Start a new session? Export first if you want to keep the current chat and evidence.")) return;
  for (let i = 0; i < 2; i++) removeImage(i);
  state.id = sessionId();
  state.startedAt = new Date().toISOString();
  state.runs = [];
  state.messages = [];
  state.selectedRun = null;
  state.selectedEvidence = null;
  Array.from($("chat-log").children).forEach((element) => { if (element.id !== "chat-welcome") element.remove(); });
  $("chat-welcome").hidden = false;
  $("evidence-empty").hidden = false;
  $("evidence-content").hidden = true;
  $("evidence-tabs").replaceChildren();
  $("evidence-image").removeAttribute("src");
  $("evidence-count").textContent = "No analysis yet";
  $("trace-overview").replaceChildren(node("p", "", "Every completed analysis includes its tool, parameters, evidence and confidence calculation."));
  $("trace-details").hidden = true;
  $("trace-details").open = false;
  $("trace-json").textContent = "";
  $("print-report").replaceChildren();
  $("query").value = "";
  renderControls();
  toast("New session started.");
}

function attachEvents() {
  $("mode").addEventListener("change", (event) => setMode(event.target.value));
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("threshold").addEventListener("input", () => { $("threshold-value").textContent = Number($("threshold").value).toFixed(2); });
  for (let index = 0; index < 2; index++) {
    const drop = $(`drop-${index}`);
    const picker = $(`file-${index}`);
    drop.addEventListener("click", (event) => { if (event.target !== picker && !state.busy && !(index === 1 && state.mode === "single")) picker.click(); });
    drop.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); if (!state.busy && !(index === 1 && state.mode === "single")) picker.click(); } });
    picker.addEventListener("change", () => { acceptFiles(picker.files, index); picker.value = ""; });
    for (const type of ["dragenter", "dragover"]) drop.addEventListener(type, (event) => { event.preventDefault(); event.stopPropagation(); if (!state.busy) drop.classList.add("dragover"); });
    drop.addEventListener("dragleave", (event) => { if (!drop.contains(event.relatedTarget)) drop.classList.remove("dragover"); });
    drop.addEventListener("drop", (event) => { event.preventDefault(); event.stopPropagation(); drop.classList.remove("dragover"); acceptFiles(event.dataTransfer.files, index); });
    document.querySelector(`[data-remove="${index}"]`).addEventListener("click", () => removeImage(index));
    document.querySelector(`[data-zoom="${index}"]`).addEventListener("click", () => showImage({ id: `image_${index ? "b" : "a"}`, label: `Image ${index ? "B" : "A"}`, data_url: state.slots[index].preview, description: state.slots[index].warnings.join(" ") }));
    $(`sensor-${index}`).addEventListener("change", () => { renderSensorFields(index); queuePreview(index); renderSuggestions(); renderControls(); });
    for (const field of ["bands", "red", "nir"]) $(`${field}-${index}`).addEventListener("input", () => queuePreview(index));
    for (const field of ["sar-scale", "date"]) $(`${field}-${index}`).addEventListener("change", () => queuePreview(index));
  }
  // Avoid accidental navigation to a raster dropped outside its target.
  for (const type of ["dragover", "drop"]) document.addEventListener(type, (event) => { if (Array.from(event.dataTransfer?.types || []).includes("Files")) event.preventDefault(); });
  $("query-form").addEventListener("submit", analyze);
  $("query").addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); $("query-form").requestSubmit(); } });
  $("cancel-analysis").addEventListener("click", () => state.controller?.abort());
  $("new-session").addEventListener("click", clearSession);
  $("new-session-mobile").addEventListener("click", clearSession);
  $("open-settings").addEventListener("click", openSettings);
  $("settings-form").addEventListener("submit", saveSettings);
  $("refresh-models").addEventListener("click", refreshModels);
  $("toggle-key").addEventListener("click", () => {
    const shown = $("api-key").type === "password";
    $("api-key").type = shown ? "text" : "password";
    $("toggle-key").textContent = shown ? "Hide" : "Show";
    $("toggle-key").setAttribute("aria-pressed", String(shown));
  });
  $("settings-dialog").addEventListener("close", () => {
    $("api-key").type = "password";
    $("api-key").value = "";
    $("workspace-access").value = "";
    $("toggle-key").textContent = "Show";
    $("toggle-key").setAttribute("aria-pressed", "false");
  });
  $("clear-key").addEventListener("click", () => {
    state.token = "";
    $("api-key").value = "";
    const cleared = writeStorage("localStorage", STORAGE.key, "");
    settingsStatus(cleared ? "Personal token cleared. A configured server token, if present, is unaffected." : "The in-memory token was cleared. Browser storage could not be modified; remove it using browser site-data settings.", !cleared);
    renderControls();
  });
  document.querySelectorAll("[data-close]").forEach((button) => button.addEventListener("click", () => $(button.dataset.close).close()));
  $("export-menu-button").addEventListener("click", () => {
    $("export-description").textContent = `${state.runs.length} completed analyses and ${state.messages.length} conversation entries, with visual evidence and execution traces.`;
    $("export-dialog").showModal();
  });
  $("download-json").addEventListener("click", downloadJSON);
  $("print-pdf").addEventListener("click", () => { printPDF().catch((error) => toast(error.message, true)); });
  $("enlarge-evidence").addEventListener("click", () => showImage(state.selectedEvidence));
  $("download-evidence").addEventListener("click", () => {
    const item = state.dialogEvidence;
    if (!item?.data_url?.startsWith("data:image/png;base64,")) return;
    const bytes = Uint8Array.from(atob(item.data_url.split(",")[1]), (character) => character.charCodeAt(0));
    downloadBlob(new Blob([bytes], { type: "image/png" }), `satquery-${item.id}.png`);
  });
  $("image-dialog").addEventListener("close", () => { $("dialog-image").removeAttribute("src"); state.dialogEvidence = null; });
  $("copy-trace").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("trace-json").textContent); toast("Execution trace copied."); }
    catch { $("trace-details").open = true; $("trace-json").focus(); const range = document.createRange(); range.selectNodeContents($("trace-json")); const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range); toast("Trace selected. Press ⌘C or Ctrl+C to copy."); }
  });
  window.addEventListener("afterprint", () => { $("print-report").replaceChildren(); });
  window.addEventListener("beforeunload", (event) => { if (state.busy) { event.preventDefault(); event.returnValue = ""; } });
}

async function initialize() {
  attachEvents();
  renderSuggestions();
  renderControls();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    state.health = await api("/api/health", { token: "", access: "", signal: controller.signal });
    state.connected = state.health.status === "ok";
    if (!state.model) state.model = state.health.default_model;
    $("connection-status").classList.add("connected");
    $("connection-label").textContent = "Backend ready";
    $("backend-version").textContent = `SATQUERY ${state.health.version}`;
  } catch (error) {
    state.connected = false;
    $("connection-status").classList.add("failed");
    $("connection-label").textContent = "Backend offline";
    toast(error.name === "AbortError" ? "The backend did not respond. Start FastAPI, then reload." : error.message, true);
  } finally { clearTimeout(timer); renderControls(); }
}

initialize().catch((error) => toast(`The workspace could not start: ${error.message}`, true));
