const state = {
  catalog: [],
  catalogIndex: new Map(),
  activeSessionId: null,
  segments: [],
  rowElements: [],
  activeSegmentIndex: -1,
};

const SEEK_STEP_SEC = 5;

const dom = {
  sessionInput: document.getElementById("sessionIdInput"),
  sessionDatalist: document.getElementById("sessionIdList"),
  loadSessionBtn: document.getElementById("loadSessionBtn"),
  statusPill: document.getElementById("statusPill"),
  video: document.getElementById("sessionVideo"),
  back5Btn: document.getElementById("back5Btn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  forward5Btn: document.getElementById("forward5Btn"),
  seekBar: document.getElementById("seekBar"),
  timeLabel: document.getElementById("timeLabel"),
  videoMeta: document.getElementById("videoMeta"),
  videoNote: document.getElementById("videoNote"),
  activeSegmentBadge: document.getElementById("activeSegmentBadge"),
  segmentsContainer: document.getElementById("segmentsContainer"),
};

function setStatus(label, tone = "neutral") {
  dom.statusPill.textContent = label;
  dom.statusPill.className = `pill ${tone}`;
}

function formatClock(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "00:00";
  }
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return `${hours.toString().padStart(2, "0")}:${minutes
      .toString()
      .padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }
  return `${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) {
    return "n/a";
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}m ${secs.toString().padStart(2, "0")}s`;
}

function buildSessionDatalist(sessions) {
  dom.sessionDatalist.innerHTML = "";
  for (const row of sessions) {
    const option = document.createElement("option");
    option.value = row.session_id;
    dom.sessionDatalist.appendChild(option);
  }
}

function formatMetricValue(value) {
  const num = Number(value);
  if (Number.isFinite(num)) {
    return num.toFixed(2);
  }
  return String(value);
}

function buildMetricChips(metrics) {
  if (!metrics || typeof metrics !== "object") {
    return "";
  }
  const entries = Object.entries(metrics);
  if (!entries.length) {
    return "";
  }
  return entries
    .map(([key, value]) => `<span class="chip metric-chip">${key}: ${formatMetricValue(value)}</span>`)
    .join("");
}

function renderSegments(segments) {
  state.rowElements = [];
  dom.segmentsContainer.innerHTML = "";

  if (!segments.length) {
    const empty = document.createElement("p");
    empty.className = "meta";
    empty.textContent = "No segments available for this session.";
    dom.segmentsContainer.appendChild(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (let i = 0; i < segments.length; i += 1) {
    const segment = segments[i];
    const row = document.createElement("button");
    row.type = "button";
    row.className = "segment-row";
    row.dataset.index = String(i);
    row.title = "Click to jump video to this segment";
    const labelChips = (segment.label_vector || [])
      .map((label) => `<span class="chip">${label}</span>`)
      .join("");
    const metricChips = buildMetricChips(segment.metrics || {});
    row.innerHTML = `
      <div class="segment-top">
        <span class="segment-id">#${segment.segment_id}</span>
        <span class="segment-time">${segment.starting_time} -> ${segment.ending_time}</span>
      </div>
      <div class="chips">
        ${labelChips}${metricChips}
      </div>
    `;
    row.addEventListener("click", () => {
      const seekTime = Math.max(0, Number(segment.starting_sec || 0));
      dom.video.currentTime = seekTime;
      dom.video.play().catch(() => {});
      setActiveSegment(i, { scroll: false });
    });
    state.rowElements.push(row);
    fragment.appendChild(row);
  }
  dom.segmentsContainer.appendChild(fragment);
}

function findSegmentIndex(currentTimeSec) {
  let left = 0;
  let right = state.segments.length - 1;
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    const segment = state.segments[mid];
    if (currentTimeSec < segment.starting_sec) {
      right = mid - 1;
      continue;
    }
    if (currentTimeSec > segment.ending_sec) {
      left = mid + 1;
      continue;
    }
    return mid;
  }
  return -1;
}

function setActiveSegment(index, options = { scroll: true }) {
  if (index === state.activeSegmentIndex) {
    return;
  }
  const prevIndex = state.activeSegmentIndex;
  state.activeSegmentIndex = index;

  if (prevIndex >= 0 && prevIndex < state.rowElements.length) {
    state.rowElements[prevIndex].classList.remove("active");
  }

  if (index < 0 || index >= state.rowElements.length) {
    dom.activeSegmentBadge.textContent = "No active segment";
    return;
  }

  const row = state.rowElements[index];
  row.classList.add("active");
  const segment = state.segments[index];
  dom.activeSegmentBadge.textContent = `Segment #${segment.segment_id} at ${segment.starting_time} -> ${segment.ending_time}`;

  if (options.scroll) {
    row.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth", container: "nearest" });
  }
}

function onVideoTimeUpdate() {
  if (!state.segments.length) {
    return;
  }
  const index = findSegmentIndex(dom.video.currentTime);
  setActiveSegment(index, { scroll: true });
}

function updatePlayPauseButton() {
  dom.playPauseBtn.textContent = dom.video.paused ? "Play" : "Pause";
}

function toggleVideoPlayback() {
  if (!dom.video.src) {
    return;
  }
  if (dom.video.paused) {
    dom.video.play().catch(() => {});
  } else {
    dom.video.pause();
  }
}

function updateSeekUi() {
  const duration = dom.video.duration;
  const current = dom.video.currentTime || 0;

  if (Number.isFinite(duration) && duration > 0) {
    const ratio = (current / duration) * 1000;
    dom.seekBar.value = String(Math.min(1000, Math.max(0, ratio)));
    dom.timeLabel.textContent = `${formatClock(current)} / ${formatClock(duration)}`;
  } else {
    dom.seekBar.value = "0";
    dom.timeLabel.textContent = `${formatClock(current)} / 00:00`;
  }
}

function setVideoControlsEnabled(enabled) {
  dom.back5Btn.disabled = !enabled;
  dom.playPauseBtn.disabled = !enabled;
  dom.forward5Btn.disabled = !enabled;
  dom.seekBar.disabled = !enabled;
}

function seekBy(secondsDelta) {
  const duration = dom.video.duration;
  const current = dom.video.currentTime || 0;
  const cap = Number.isFinite(duration) && duration > 0 ? duration : current + Math.abs(secondsDelta);
  const next = Math.min(Math.max(current + secondsDelta, 0), cap);
  dom.video.currentTime = next;
  onVideoTimeUpdate();
  updateSeekUi();
}

function updateVideoMeta(sessionPayload) {
  const metadata = sessionPayload.metadata || {};
  const turnStats = metadata.turn_event_stats || {};
  const line1 = `Session ${sessionPayload.session_id} | ${sessionPayload.segments.length} segments`;
  const line2 = `Duration ${formatDuration(metadata.source_duration_sec || 0)} | Turns ${
    turnStats.num_turn_events || 0
  } | Sharp ${turnStats.sharp_turn_events || 0}`;
  dom.videoMeta.textContent = `${line1} | ${line2}`;
}

function updateVideoSource(sessionPayload) {
  dom.video.pause();
  dom.video.removeAttribute("src");
  dom.video.load();
  updatePlayPauseButton();
  updateSeekUi();

  if (!sessionPayload.video_url) {
    dom.videoNote.textContent =
      "No video mapped for this session. Use --video-root or --video-map when starting the server.";
    setVideoControlsEnabled(false);
    return;
  }

  setVideoControlsEnabled(true);
  dom.video.src = sessionPayload.video_url;
  const sizeMb = Number(sessionPayload.video_size_bytes || 0) / (1024 * 1024);
  const sizeText = Number.isFinite(sizeMb) ? `${sizeMb.toFixed(1)} MB` : "unknown size";
  dom.videoNote.textContent = `Video file: ${
    sessionPayload.video_filename || "mapped"
  } (${sizeText}) | Seek step: ${SEEK_STEP_SEC}s`;
}

async function fetchCatalog() {
  const response = await fetch("/api/sessions", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load session catalog (${response.status})`);
  }
  return response.json();
}

async function loadSession(sessionId) {
  if (!sessionId) {
    setStatus("Please enter a session id", "warn");
    return;
  }
  setStatus(`Loading session ${sessionId}...`, "neutral");

  const response = await fetch(`/api/session/${encodeURIComponent(sessionId)}`, {
    cache: "no-store",
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const message = payload.error || `Session fetch failed (${response.status})`;
    throw new Error(message);
  }

  const payload = await response.json();
  state.activeSessionId = payload.session_id;
  state.segments = payload.segments || [];
  state.activeSegmentIndex = -1;

  renderSegments(state.segments);
  updateVideoMeta(payload);
  updateVideoSource(payload);
  dom.sessionInput.value = payload.session_id;

  setStatus(
    `Loaded ${payload.session_id} (${payload.segments.length} segments)`,
    "success",
  );
}

function pickInitialSession(catalogPayload) {
  const featured = catalogPayload.featured_sessions || [];
  if (featured.length) {
    return featured[0];
  }
  const withVideo = (catalogPayload.sessions || []).find((item) => item.has_video);
  if (withVideo) {
    return withVideo.session_id;
  }
  return catalogPayload.sessions?.[0]?.session_id || "";
}

async function initialize() {
  try {
    const payload = await fetchCatalog();
    state.catalog = payload.sessions || [];
    state.catalogIndex = new Map(state.catalog.map((row) => [row.session_id, row]));

    buildSessionDatalist(state.catalog);

    const mapped = payload.video_mapped_sessions || 0;
    setStatus(
      `${payload.total_sessions || state.catalog.length} sessions indexed, ${mapped} with video`,
      mapped > 0 ? "success" : "warn",
    );

    const initialSession = pickInitialSession(payload);
    if (initialSession) {
      await loadSession(initialSession);
    }
  } catch (error) {
    setStatus(error.message, "error");
  }
}

dom.loadSessionBtn.addEventListener("click", async () => {
  try {
    await loadSession(dom.sessionInput.value.trim());
  } catch (error) {
    setStatus(error.message, "error");
  }
});

dom.sessionInput.addEventListener("keydown", async (event) => {
  if (event.key !== "Enter") {
    return;
  }
  event.preventDefault();
  try {
    await loadSession(dom.sessionInput.value.trim());
  } catch (error) {
    setStatus(error.message, "error");
  }
});

dom.back5Btn.addEventListener("click", () => seekBy(-SEEK_STEP_SEC));
dom.forward5Btn.addEventListener("click", () => seekBy(SEEK_STEP_SEC));

dom.playPauseBtn.addEventListener("click", toggleVideoPlayback);

dom.seekBar.addEventListener("input", () => {
  const duration = dom.video.duration;
  if (!Number.isFinite(duration) || duration <= 0) {
    return;
  }
  const ratio = Number(dom.seekBar.value) / 1000;
  dom.video.currentTime = Math.min(Math.max(ratio * duration, 0), duration);
  onVideoTimeUpdate();
  updateSeekUi();
});

document.addEventListener("keydown", (event) => {
  const tag = (event.target && event.target.tagName) || "";
  const isTextInput =
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    Boolean(event.target && event.target.isContentEditable);
  if (isTextInput) {
    return;
  }
  if (!dom.video.src) {
    return;
  }

  if (event.code === "Space") {
    event.preventDefault();
    toggleVideoPlayback();
    return;
  }

  if (event.key === "ArrowRight" || event.key.toLowerCase() === "l") {
    event.preventDefault();
    seekBy(SEEK_STEP_SEC);
    return;
  }
  if (event.key === "ArrowLeft" || event.key.toLowerCase() === "j") {
    event.preventDefault();
    seekBy(-SEEK_STEP_SEC);
    return;
  }
});

dom.video.addEventListener("timeupdate", () => {
  onVideoTimeUpdate();
  updateSeekUi();
});
dom.video.addEventListener("seeked", onVideoTimeUpdate);
dom.video.addEventListener("loadedmetadata", updateSeekUi);
dom.video.addEventListener("durationchange", updateSeekUi);
dom.video.addEventListener("play", updatePlayPauseButton);
dom.video.addEventListener("pause", updatePlayPauseButton);
dom.video.addEventListener("ended", updatePlayPauseButton);

setVideoControlsEnabled(false);
updatePlayPauseButton();
updateSeekUi();

initialize();
