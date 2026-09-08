// ===== DOM REFS =====
const videoSubject = document.getElementById("videoSubject");
const aiModel = document.getElementById("aiModel");
const voice = document.getElementById("voice");
const songFiles = document.getElementById("songFiles");
const paragraphNumber = document.getElementById("paragraphNumber");
const youtubeToggle = document.getElementById("youtubeUploadToggle");
const useMusicToggle = document.getElementById("useMusicToggle");
const customPrompt = document.getElementById("customPrompt");
const generateButton = document.getElementById("generateButton");
const cancelButton = document.getElementById("cancelButton");
const statusArea = document.getElementById("statusArea");
const colorDot = document.getElementById("colorDot");
const subtitlesColor = document.getElementById("subtitlesColor");
const logViewer = document.getElementById("logViewer");
const logViewerBody = document.getElementById("logViewerBody");
const logClearBtn = document.getElementById("logClearBtn");
const logAutoScroll = document.getElementById("logAutoScroll");
const logCopyBtn = document.getElementById("logCopyBtn");
const logExpandBtn = document.getElementById("logExpandBtn");
const generationStatus = document.getElementById("generationStatus");
const jobStateBadge = document.getElementById("jobStateBadge");
const jobLabel = document.getElementById("jobLabel");
const elapsedTime = document.getElementById("elapsedTime");
const jobSummary = document.getElementById("jobSummary");
const jobSummaryTitle = document.getElementById("jobSummaryTitle");
const jobSummaryMessage = document.getElementById("jobSummaryMessage");
const backendHost = window.location.hostname || "localhost";
const backendProtocol = window.location.protocol || "http:";
const API_BASE_URL = `${backendProtocol}//${backendHost}:8080`;
const API_FALLBACK_URL = `http://${backendHost}:8080`;

let activeJobId = null;
let pollHandle = null;
let lastEventId = 0;
let pollSession = 0;
let pollInFlight = false;
let connectionLost = false;
let lastPollingWarning = 0;
let isGenerating = false;
let elapsedHandle = null;
let startedAt = null;
let logEntries = [];
let focusBeforeExpand = null;

// ===== API HELPERS =====
async function apiRequest(path, options = {}) {
  const endpoint = path.startsWith("/") ? path : `/${path}`;

  async function request(baseUrl) {
    const response = await fetch(`${baseUrl}${endpoint}`, options);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.message || `Request failed with status ${response.status}`);
    }
    return data;
  }

  try {
    return await request(API_BASE_URL);
  } catch (firstError) {
    if (API_BASE_URL !== API_FALLBACK_URL) {
      return request(API_FALLBACK_URL);
    }
    throw firstError;
  }
}

function setModelOptions(models, preferredModel) {
  aiModel.replaceChildren();

  models.forEach((modelName) => {
    const option = document.createElement("option");
    option.value = modelName;
    option.textContent = modelName;
    aiModel.appendChild(option);
  });

  if (preferredModel && models.includes(preferredModel)) {
    aiModel.value = preferredModel;
  } else if (models.length > 0) {
    aiModel.value = models[0];
  }
}

async function loadOllamaModels() {
  const fallbackModel = "llama3.1:8b";

  try {
    const data = await apiRequest("/api/models", {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    const models = Array.isArray(data.models)
      ? data.models.filter((item) => typeof item === "string" && item.trim())
      : [];
    const defaultModel =
      typeof data.default === "string" && data.default.trim()
        ? data.default.trim()
        : fallbackModel;

    if (data.status && data.status !== "success" && data.message) {
      showToast(data.message, "error");
    }

    if (models.length === 0) {
      setModelOptions([defaultModel], defaultModel);
      showToast("No Ollama models found. Pull one with: ollama pull llama3.1:8b", "error");
      return;
    }

    setModelOptions(models, defaultModel);
  } catch {
    setModelOptions([fallbackModel], fallbackModel);
    showToast("Could not load Ollama models. Is backend/Ollama running?", "error");
  }
}

// ===== TOAST NOTIFICATIONS =====
function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  const dot = document.createElement("span");
  dot.className = "toast-dot";
  const text = document.createElement("span");
  text.className = "toast-msg";
  text.textContent = message;
  const close = document.createElement("button");
  close.className = "toast-close";
  close.type = "button";
  close.setAttribute("aria-label", "Close notification");
  close.textContent = "×";
  toast.append(dot, text, close);
  close.addEventListener("click", () => {
    dismissToast(toast);
  });
  container.appendChild(toast);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add("show"));
  });

  setTimeout(() => dismissToast(toast), 5000);
}

function dismissToast(toast) {
  toast.classList.remove("show");
  toast.addEventListener("transitionend", () => toast.remove(), { once: true });
}

// ===== COLOR DOT =====
function updateColorDot() {
  if (colorDot && subtitlesColor) {
    colorDot.style.backgroundColor = subtitlesColor.value;
  }
}
updateColorDot();
subtitlesColor.addEventListener("change", updateColorDot);

// ===== LIVE JOB OUTPUT =====
function formatTimestamp(ts) {
  const d = new Date((ts ?? Date.now() / 1000) * 1000);
  return d.toLocaleTimeString("en-GB", { hour12: false });
}

function appendLogEntry(entry) {
  const level = ["info", "success", "warning", "error"].includes(entry.level) ? entry.level : "info";
  const timestamp = formatTimestamp(entry.timestamp);
  const message = String(entry.message ?? "");
  logEntries.push({ timestamp, level, message });
  const previousScroll = logViewerBody.scrollTop;
  const row = document.createElement("div");
  row.className = "log-entry";
  row.dataset.level = level;

  const time = document.createElement("span");
  time.className = "log-time";
  time.textContent = timestamp;

  const severity = document.createElement("span");
  severity.className = "log-level";
  severity.textContent = level.toUpperCase();

  const msg = document.createElement("span");
  msg.className = `log-msg log-${level}`;
  msg.textContent = message;

  row.appendChild(time);
  row.appendChild(severity);
  row.appendChild(msg);
  logViewerBody.appendChild(row);

  logViewerBody.scrollTop = logAutoScroll.checked ? logViewerBody.scrollHeight : previousScroll;
}

function clearLogs() {
  logEntries = [];
  logViewerBody.replaceChildren();
}

function updateElapsed() {
  const seconds = startedAt === null ? 0 : Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
  elapsedTime.textContent = `Elapsed ${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function setJobStatus(state) {
  const labels = {
    queued: "● Waiting in queue...", running: "● Generating video...",
    completed: "✓ Video generated successfully", failed: "✕ Generation failed",
    cancelled: "■ Generation cancelled",
  };
  generationStatus.textContent = labels[state] || "Preparing generation...";
  jobStateBadge.textContent = state.toUpperCase();
  jobStateBadge.dataset.state = state;
}

function finishGeneration(state, message) {
  const level = state === "failed" ? "error" : state === "cancelled" ? "warning" : "success";
  if (!logEntries.some((entry) => entry.message.includes(message))) {
    appendLogEntry({ level, message });
  }
  setJobStatus(state);
  jobSummary.dataset.state = state;
  jobSummaryTitle.textContent = `GENERATION ${state.toUpperCase()}`;
  jobSummaryMessage.textContent = message;
  jobSummary.classList.remove("hidden");
  setGeneratingState(false);
  showToast(message, level);
}

async function refreshEvents(jobId, session) {
  const result = await apiRequest(`/api/jobs/${jobId}/events?after=${lastEventId}`, {
    method: "GET", headers: { Accept: "application/json" },
  });
  if (session !== pollSession || jobId !== activeJobId) return 0;
  if (result.status !== "success" || !Array.isArray(result.events)) {
    throw new Error(result.message || "Invalid job events response.");
  }
  result.events.forEach((event) => {
    if (event.id > lastEventId) {
      appendLogEntry(event);
      lastEventId = event.id;
    }
  });
  return result.events.length;
}

async function pollJob() {
  if (!activeJobId || pollInFlight) return;
  const jobId = activeJobId;
  const session = pollSession;
  pollInFlight = true;

  try {
    await refreshEvents(jobId, session);
    if (session !== pollSession) return;
    const jobResult = await apiRequest(`/api/jobs/${jobId}`, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });
    if (session !== pollSession) return;
    const state = jobResult?.job?.state;
    if (jobResult.status !== "success" || !["queued", "running", "completed", "failed", "cancelled"].includes(state)) {
      throw new Error(jobResult.message || "Invalid job status response.");
    }
    if (["completed", "failed", "cancelled"].includes(state)) {
      // Fetch the tail written between the event request and terminal status.
      // The existing API returns pages of at most 200 events.
      while (await refreshEvents(jobId, session) === 200) {
        if (session !== pollSession) return;
      }
      if (session !== pollSession) return;
    }
    if (connectionLost) {
      appendLogEntry({ level: "success", message: "[UI] Job connection restored." });
      connectionLost = false;
    }
    setJobStatus(state);
    if (state === "completed") {
      finishGeneration(state, "Video generated successfully.");
    } else if (state === "failed") {
      finishGeneration(state, jobResult.job.errorMessage || "Generation failed.");
    } else if (state === "cancelled") {
      finishGeneration(state, "Generation cancelled.");
    }
  } catch (error) {
    if (session !== pollSession) return;
    if (!connectionLost || Date.now() - lastPollingWarning >= 30000) {
      appendLogEntry({ level: "warning", message: `[UI] Could not refresh job status. Retrying...\n${error.message}` });
      lastPollingWarning = Date.now();
    }
    connectionLost = true;
  } finally {
    if (session === pollSession) pollInFlight = false;
  }
}

function startJobPolling(jobId) {
  stopJobPolling();
  activeJobId = jobId;
  lastEventId = 0;
  connectionLost = false;
  lastPollingWarning = 0;
  jobLabel.textContent = `Job ${jobId.slice(0, 8)}`;
  jobLabel.title = jobId;
  setJobStatus("queued");
  cancelButton.disabled = false;
  startedAt = Date.now();
  updateElapsed();
  elapsedHandle = setInterval(updateElapsed, 1000);
  pollHandle = setInterval(pollJob, 1200);
  pollJob();
}

function stopJobPolling() {
  pollSession += 1;
  pollInFlight = false;
  if (pollHandle) {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

// ===== GENERATE / CANCEL =====
function setGeneratingState(active) {
  isGenerating = active;
  if (active) {
    generateButton.classList.add("hidden");
    cancelButton.classList.remove("hidden");
    statusArea.classList.add("active");
    cancelButton.disabled = true; // Cancellation becomes available after a job ID arrives.
  } else {
    stopJobPolling();
    activeJobId = null;
    generateButton.classList.remove("hidden");
    cancelButton.classList.add("hidden");
    statusArea.classList.remove("active");
    generateButton.disabled = false;
    cancelButton.disabled = true;
    clearInterval(elapsedHandle);
    elapsedHandle = null;
    updateElapsed();
  }
}

function cancelGeneration() {
  if (!activeJobId || cancelButton.disabled) return;
  const jobId = activeJobId;
  const targetPath = `/api/jobs/${jobId}/cancel`;
  cancelButton.disabled = true;

  apiRequest(targetPath, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  })
    .then((data) => {
      if (activeJobId !== jobId) return;
      if (data.status !== "success") throw new Error(data.message || "Cancellation request failed.");
      appendLogEntry({ level: "info", message: data.message || "Cancellation requested." });
      showToast(data.message || "Cancellation requested.", "info");
    })
    .catch((error) => {
      if (activeJobId !== jobId) return;
      const message = `[UI] Failed to request cancellation: ${error.message}`;
      appendLogEntry({ level: "warning", message });
      showToast(message, "error");
      cancelButton.disabled = false;
    });
}

async function uploadSongs() {
  const files = songFiles.files;
  if (!files || files.length === 0) return true;

  const mp3s = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".mp3"));
  if (mp3s.length === 0) {
    showToast("No MP3 files found in the selected folder.", "error");
    return false;
  }

  const formData = new FormData();
  mp3s.forEach((file) => formData.append("songs", file));

  try {
    await apiRequest("/api/upload-songs", {
      method: "POST",
      body: formData,
    });
    return true;
  } catch {
    showToast("Failed to upload songs.", "error");
    return false;
  }
}

async function generateVideo() {
  if (isGenerating) return;
  const subject = videoSubject.value.trim();
  if (!subject) {
    showToast("Please enter a video subject.", "error");
    videoSubject.focus();
    return;
  }

  generateButton.disabled = true;
  setGeneratingState(true);

  // Clear previous log entries
  clearLogs();
  jobSummary.classList.add("hidden");
  jobLabel.textContent = "Submitting job...";
  jobLabel.removeAttribute("title");
  jobStateBadge.textContent = "QUEUED";
  jobStateBadge.dataset.state = "queued";
  generationStatus.textContent = "Preparing generation...";
  startedAt = null;
  updateElapsed();

  // Upload songs first if a folder was selected
  if (useMusicToggle.checked && songFiles.files.length > 0) {
    const uploaded = await uploadSongs();
    if (!uploaded) {
      finishGeneration("failed", "Could not upload songs. Generation was not started.");
      return;
    }
  }

  const data = {
    videoSubject: subject,
    aiModel: aiModel.value || "llama3.1:8b",
    voice: voice.value,
    paragraphNumber: paragraphNumber.value,
    automateYoutubeUpload: youtubeToggle.checked,
    useMusic: useMusicToggle.checked,
    threads: document.getElementById("threads").value,
    subtitlesPosition: document.getElementById("subtitlesPosition").value,
    aspectRatio: document.getElementById("aspectRatio").value,
    minDuration: document.getElementById("minDuration").value,
    customPrompt: customPrompt.value,
    color: subtitlesColor.value,
  };

  try {
    const result = await apiRequest("/api/generate", {
      method: "POST",
      body: JSON.stringify(data),
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    });

    if (result.status === "success") {
      if (!result.jobId) {
        finishGeneration("failed", "Generation queued, but no job ID was returned.");
        return;
      }
      startJobPolling(result.jobId);
    } else {
      finishGeneration("failed", result.message || "Generation request failed.");
    }
  } catch (error) {
    finishGeneration("failed", `Could not submit generation: ${error.message}`);
  }
}

generateButton.addEventListener("click", generateVideo);
cancelButton.addEventListener("click", cancelGeneration);

videoSubject.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    generateVideo();
  }
});

// ===== LOG CLEAR BUTTON =====
logClearBtn.addEventListener("click", clearLogs);
logAutoScroll.addEventListener("change", () => {
  if (logAutoScroll.checked) logViewerBody.scrollTop = logViewerBody.scrollHeight;
});
logViewerBody.addEventListener("scroll", () => {
  if (logViewerBody.scrollHeight - logViewerBody.scrollTop - logViewerBody.clientHeight > 24) {
    logAutoScroll.checked = false;
  }
});

logCopyBtn.addEventListener("click", async () => {
  const text = logEntries.map((entry) => `[${entry.timestamp}] [${entry.level.toUpperCase()}] ${entry.message}`).join("\n");
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      // Local HTTP deployments may not expose the Clipboard API.
      const field = document.createElement("textarea");
      field.value = text;
      field.style.cssText = "position:fixed;opacity:0;";
      logViewer.appendChild(field);
      field.select();
      try {
        if (!document.execCommand("copy")) throw new Error("Clipboard unavailable");
      } finally {
        field.remove();
        logCopyBtn.focus();
      }
    }
    showToast("Logs copied.", "success");
  } catch {
    showToast("Could not copy logs. Select the log text and copy it manually.", "error");
  }
});

function setLogsExpanded(expanded) {
  if (expanded) focusBeforeExpand = document.activeElement;
  logViewer.classList.toggle("expanded", expanded);
  document.body.classList.toggle("logs-expanded", expanded);
  logExpandBtn.setAttribute("aria-expanded", String(expanded));
  logExpandBtn.textContent = expanded ? "Collapse" : "Expand";
  for (const element of document.querySelectorAll(".header, #generationForm, #statusArea, .footer")) {
    element.inert = expanded;
  }
  if (expanded) {
    logViewer.setAttribute("role", "dialog");
    logViewer.setAttribute("aria-modal", "true");
    logExpandBtn.focus();
  } else {
    logViewer.removeAttribute("role");
    logViewer.removeAttribute("aria-modal");
    focusBeforeExpand?.focus();
  }
}
logExpandBtn.addEventListener("click", () => setLogsExpanded(!logViewer.classList.contains("expanded")));
document.addEventListener("keydown", (event) => {
  if (!logViewer.classList.contains("expanded")) return;
  if (event.key === "Escape") {
    event.preventDefault();
    setLogsExpanded(false);
  } else if (event.key === "Tab") {
    const controls = Array.from(logViewer.querySelectorAll("button, input, [tabindex='0']"));
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault(); last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault(); first.focus();
    }
  }
});

// ===== LOCAL STORAGE PERSISTENCE =====
const SETTINGS_STORAGE_KEY = "moneyPrinter.settings.v1";
// Explicit allowlist: never include subject, file inputs, paths, or credentials.
const settingsControls = {
  model: aiModel,
  voice,
  aspectRatio: document.getElementById("aspectRatio"),
  subtitlePosition: document.getElementById("subtitlesPosition"),
  subtitleColor: subtitlesColor,
  threads: document.getElementById("threads"),
  paragraphs: paragraphNumber,
  minDuration: document.getElementById("minDuration"),
  uploadYoutube: youtubeToggle,
  useMusic: useMusicToggle,
  reuseChoices: document.getElementById("reuseChoicesToggle"),
  customPrompt,
};
let savedSettings = {};
let defaultSettings = {};
let modelOptionsReady = false;
let restoringSettings = false;
let storageWarningShown = false;

function warnSettingsStorage() {
  if (storageWarningShown) return;
  storageWarningShown = true;
  showToast("Browser settings storage is unavailable or invalid. Preferences may not survive a reload.", "error");
}

function readSavedSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem(SETTINGS_STORAGE_KEY) || "{}");
    if (!stored || typeof stored !== "object" || Array.isArray(stored)) return {};
    return Object.fromEntries(Object.keys(settingsControls)
      .filter((key) => Object.hasOwn(stored, key))
      .map((key) => [key, stored[key]]));
  } catch {
    warnSettingsStorage();
    return {};
  }
}

function validSetting(control, value) {
  if (control.type === "checkbox") return typeof value === "boolean";
  if (control.type === "number") {
    return typeof value === "number" && Number.isFinite(value) && Number.isInteger(value)
      && (control.min === "" || value >= Number(control.min))
      && (control.max === "" || value <= Number(control.max));
  }
  if (typeof value !== "string") return false;
  if (control.tagName === "SELECT") {
    return value !== "" && Array.from(control.options).some((option) =>
      option.value === value && !option.disabled && !option.parentElement.disabled);
  }
  return control === customPrompt;
}

function readSetting(control, defaults = false) {
  if (control.type === "checkbox") return defaults ? control.defaultChecked : control.checked;
  let value = defaults ? control.defaultValue : control.value;
  if (defaults && control.tagName === "SELECT") {
    value = (Array.from(control.options).find((option) => option.defaultSelected) || control.options[0])?.value;
  }
  if (control.type === "number") return value === "" ? undefined : Number(value);
  return value;
}

function restoreSettings(settings) {
  restoringSettings = true;
  try {
    for (const [key, control] of Object.entries(settingsControls)) {
      if (key === "model" && !modelOptionsReady) continue;
      if (!validSetting(control, settings[key])) continue;
      if (control.type === "checkbox") control.checked = settings[key];
      else control.value = settings[key];
      // Styled native selects render their selected option's label automatically.
      // Notify dependent UI (e.g. the color dot) without re-saving during restore.
      control.dispatchEvent(new Event("change", { bubbles: true }));
    }
  } finally {
    restoringSettings = false;
  }
}

function saveSettings() {
  if (restoringSettings) return;
  const settings = {};
  for (const [key, control] of Object.entries(settingsControls)) {
    if (key === "model" && !modelOptionsReady) {
      // Editing another field while models load must not erase the saved model.
      if (typeof savedSettings.model === "string") settings.model = savedSettings.model;
      continue;
    }
    const value = readSetting(control);
    if (validSetting(control, value)) settings[key] = value;
  }
  savedSettings = settings;
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    warnSettingsStorage();
  }
}

function resetSettings() {
  savedSettings = {};
  try {
    localStorage.removeItem(SETTINGS_STORAGE_KEY);
  } catch {
    warnSettingsStorage();
  }
  restoreSettings(defaultSettings);
}

document.addEventListener("DOMContentLoaded", async () => {
  defaultSettings = Object.fromEntries(Object.entries(settingsControls)
    .map(([key, control]) => [key, readSetting(control, true)]));
  savedSettings = readSavedSettings();
  restoreSettings(defaultSettings);
  restoreSettings(savedSettings);

  for (const control of Object.values(settingsControls)) {
    control.addEventListener("change", saveSettings);
    if (control.type === "number" || control === customPrompt) {
      control.addEventListener("input", saveSettings);
    }
  }
  document.getElementById("resetSettingsButton").addEventListener("click", resetSettings);

  await loadOllamaModels();
  defaultSettings.model = aiModel.value;
  modelOptionsReady = true;
  restoreSettings({ model: savedSettings.model });
});
