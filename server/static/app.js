/**
 * app.js — Training data collection UI.
 * Non-blocking queue workflow: record words, review/save later.
 * Records audio as WAV alongside accelerometer data.
 */

// ── DOM refs ───────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const $dot = $("connection-dot");
const $connText = $("connection-text");
const $sampleRate = $("sample-rate");
const $totalSamples = $("total-samples");
const $recordBtn = $("record-btn");
const $recordLabel = $recordBtn.querySelector(".record-label");
const $recordSub = $recordBtn.querySelector(".record-sub");
const $newlineBtn = $("newline-btn");
const $transcript = $("transcript");
const $transcriptLineNum = $("transcript-line-num");
const $speechStatus = $("speech-status");
const $speechFeedback = $("speech-feedback");
const $speechText = $("speech-text");
const $recordingTimer = $("recording-timer");
const $queueSection = $("queue-section");
const $queueCount = $("queue-count");
const $queueList = $("queue-list");
const $btnSaveAll = $("btn-save-all");
const $btnDiscardAll = $("btn-discard-all");
const $historyList = $("history-list");
const $wordCounts = $("word-counts");
const $datasetTotal = $("dataset-total");
const $sessionBtn = $("session-btn");
const $sessionStatus = $("session-status");

// ── State ──────────────────────────────────────────────
let ws = null;
let connected = false;
let isRecording = false;
let sessionActive = false;
let sessionTimer = null;
let sessionStartTime = null;
let sessionNeedsFinalize = false;
let speechResult = null;
let recordingStartTime = null;
let timerInterval = null;
let currentLine = 1;
let lastWavBlob = null;

const MAX_RECORD_SEC = 10;
const pendingQueue = [];
const savedHistory = [];

function samplesToDeltas(samples, timestamps) {
    const deltas = [];
    for (let i = 1; i < samples.length; i++) {
        deltas.push({
            x: samples[i][0] - samples[i - 1][0],
            y: samples[i][1] - samples[i - 1][1],
            z: samples[i][2] - samples[i - 1][2],
            t: timestamps[i],
        });
    }
    return deltas;
}

// ── Audio Recording (WAV via Web Audio API) ────────────
let audioStream = null;
let audioCtx = null;
let audioSource = null;
let audioProcessor = null;
let audioChunks = [];
let audioReady = false;

async function initAudio() {
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        audioSource = audioCtx.createMediaStreamSource(audioStream);
        audioProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
        audioProcessor.onaudioprocess = (e) => {
            if (isRecording) {
                audioChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
            }
        };
        audioSource.connect(audioProcessor);
        audioProcessor.connect(audioCtx.destination);
        audioReady = true;
    } catch (err) {
        console.warn("Audio recording unavailable:", err);
    }
}

function startAudioRecording() {
    audioChunks = [];
    if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
}

function stopAudioRecording() {
    if (!audioReady || audioChunks.length === 0) return null;
    const totalLen = audioChunks.reduce((sum, c) => sum + c.length, 0);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of audioChunks) { merged.set(chunk, offset); offset += chunk.length; }
    audioChunks = [];
    return encodeWAV(merged, audioCtx.sampleRate);
}

function encodeWAV(samples, sampleRate) {
    const bitsPerSample = 16;
    const dataSize = samples.length * 2;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeStr(off, str) {
        for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
    }

    writeStr(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, bitsPerSample, true);
    writeStr(36, "data");
    view.setUint32(40, dataSize, true);

    let pos = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(pos, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        pos += 2;
    }
    return new Blob([buffer], { type: "audio/wav" });
}

initAudio();

// ── Charts ─────────────────────────────────────────────
const liveChart = new AccelChart(document.getElementById("live-chart"));
let prevRaw = null;

(function animLoop() {
    liveChart.draw();
    requestAnimationFrame(animLoop);
})();

// ── WebSocket ──────────────────────────────────────────
function connectWS() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onmessage = (evt) => {
        const data = JSON.parse(evt.data);
        switch (data.type) {
            case "accel":
                if (prevRaw) {
                    liveChart.push({
                        x: data.x - prevRaw.x, y: data.y - prevRaw.y,
                        z: data.z - prevRaw.z, t: data.t,
                    });
                }
                prevRaw = { x: data.x, y: data.y, z: data.z };
                break;
            case "status":
                connected = data.connected;
                $dot.className = "dot " + (connected ? "connected" : "disconnected");
                $connText.textContent = connected ? `Connected: ${data.port}` : "Disconnected";
                $sampleRate.textContent = connected ? `${data.sample_rate_hz} Hz` : "\u2014 Hz";
                updateRecordButton();
                break;
            case "stats":
            case "sample_saved":
                updateDatasetUI(data);
                break;
            case "recording_stopped":
                onRecordingStopped(data);
                break;
            case "session_started":
                if (data.session_file) $sessionStatus.textContent = data.session_file;
                break;
            case "session_stopped":
                $sessionStatus.textContent = `Stopped \u2014 ${data.duration_s}s recorded. Save words then click Save All to finalize.`;
                break;
            case "session_finalized":
                $sessionStatus.textContent = `Session saved: ${data.file}`;
                setTimeout(() => { $sessionStatus.textContent = ""; }, 8000);
                break;
        }
    };

    ws.onclose = () => {
        connected = false;
        $dot.className = "dot disconnected";
        $connText.textContent = "Disconnected \u2014 reconnecting...";
        updateRecordButton();
        setTimeout(connectWS, 2000);
    };
    ws.onerror = () => ws.close();
}
connectWS();

// ── Session ────────────────────────────────────────────
$sessionBtn.addEventListener("click", () => {
    if (!connected) return;
    if (!sessionActive) {
        sessionActive = true;
        sessionNeedsFinalize = false;
        sessionStartTime = performance.now();
        $sessionBtn.textContent = "Stop Session";
        $sessionBtn.className = "session-btn-active";
        $sessionStatus.textContent = "0:00";
        sessionTimer = setInterval(() => {
            const sec = Math.floor((performance.now() - sessionStartTime) / 1000);
            $sessionStatus.textContent = `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
        }, 1000);
        ws.send(JSON.stringify({ type: "start_session" }));
    } else {
        sessionActive = false;
        sessionNeedsFinalize = true;
        clearInterval(sessionTimer);
        $sessionBtn.textContent = "Start Session";
        $sessionBtn.className = "session-btn-idle";
        ws.send(JSON.stringify({ type: "stop_session" }));
    }
});

function finalizeSession() {
    if (!sessionNeedsFinalize) return;
    sessionNeedsFinalize = false;
    const words = savedHistory.slice().reverse().map(h => ({
        sample_id: h.id, word: h.word, line: h.line || 1,
    }));
    ws.send(JSON.stringify({ type: "finalize_session", words }));
}

// ── Record button ──────────────────────────────────────
function updateRecordButton() {
    $recordBtn.disabled = !connected;
}

function startRecording(evt) {
    if (!connected || isRecording) return;
    if (evt.pointerId !== undefined) $recordBtn.setPointerCapture(evt.pointerId);

    isRecording = true;
    recordingStartTime = performance.now();
    $recordBtn.classList.add("recording");
    $recordLabel.textContent = "RECORDING...";
    $recordSub.textContent = "release when done";
    $speechFeedback.classList.remove("hidden");
    $speechText.textContent = "Listening...";
    $recordingTimer.classList.remove("hidden");
    $recordingTimer.textContent = "0.0s";
    liveChart.recording = true;

    timerInterval = setInterval(() => {
        const elapsed = (performance.now() - recordingStartTime) / 1000;
        $recordingTimer.textContent = elapsed.toFixed(1) + "s";
        if (elapsed >= MAX_RECORD_SEC) stopRecording();
    }, 100);

    startSpeech();
    startAudioRecording();
    ws.send(JSON.stringify({ type: "start_recording" }));
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    clearInterval(timerInterval);
    $recordBtn.classList.remove("recording");
    $recordLabel.textContent = "HOLD TO RECORD";
    $recordSub.textContent = "say the word aloud while writing";
    $recordingTimer.classList.add("hidden");
    liveChart.recording = false;
    stopSpeech();
    lastWavBlob = stopAudioRecording();
    ws.send(JSON.stringify({ type: "stop_recording" }));
}

$recordBtn.addEventListener("pointerdown", (e) => { e.preventDefault(); startRecording(e); });
$recordBtn.addEventListener("pointerup", (e) => { e.preventDefault(); stopRecording(); });
$recordBtn.addEventListener("pointercancel", () => stopRecording());

function isInputFocused() {
    const el = document.activeElement;
    return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA");
}

document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && !isInputFocused()) { e.preventDefault(); startRecording(e); }
});
document.addEventListener("keyup", (e) => {
    if (e.code === "Space" && !isInputFocused()) { e.preventDefault(); stopRecording(); }
});

// ── Speech Recognition ─────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (!SpeechRecognition) {
    $speechStatus.textContent = "Speech API not available \u2014 type labels manually";
    $speechStatus.className = "speech-warn";
} else {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.maxAlternatives = 3;
    $speechStatus.textContent = "Speech recognition ready";
    $speechStatus.className = "speech-ok";
    setTimeout(() => { $speechStatus.style.opacity = "0"; }, 4000);

    recognition.onresult = (event) => {
        const result = event.results[0];
        const transcript = result[0].transcript.trim().toLowerCase();
        $speechText.textContent = result.isFinal ? `"${transcript}"` : `${transcript}...`;
        if (result.isFinal) speechResult = transcript;
    };
    recognition.onerror = (event) => {
        if (event.error === "not-allowed") {
            $speechText.textContent = "Mic blocked \u2014 check browser permissions";
            $speechStatus.textContent = "Microphone denied \u2014 click lock icon to allow";
            $speechStatus.className = "speech-warn";
            $speechStatus.style.opacity = "1";
        } else if (event.error === "no-speech") {
            $speechText.textContent = "No speech detected";
        } else if (event.error !== "aborted") {
            $speechText.textContent = `Speech error: ${event.error}`;
        }
    };
}

function startSpeech() {
    speechResult = null;
    if (recognition) try { recognition.start(); } catch (_) {}
}
function stopSpeech() {
    if (recognition) try { recognition.stop(); } catch (_) {}
}

// ── New Line ───────────────────────────────────────────
$newlineBtn.addEventListener("click", () => {
    currentLine++;
    pendingQueue.push({ type: "newline", line: currentLine });
    $transcriptLineNum.textContent = `Line ${currentLine}`;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "mark_newline", line: currentLine }));
    }
    renderQueue();
    renderTranscript();
});

// ── Recording stopped -> queue ─────────────────────────
function onRecordingStopped(data) {
    $speechFeedback.classList.add("hidden");
    pendingQueue.push({
        type: "word",
        sample_id: data.sample_id,
        word: speechResult || "",
        line: currentLine,
        samples: data.samples,
        timestamps: data.timestamps,
        duration_s: data.duration_s,
        num_samples: data.num_samples,
        recorded_at: data.recorded_at,
        wavBlob: lastWavBlob,
    });
    speechResult = null;
    lastWavBlob = null;
    renderQueue();
    renderTranscript();
}

// ── Transcript ─────────────────────────────────────────
function renderTranscript() {
    const lines = {};
    let wordIdx = 0;
    for (const entry of pendingQueue) {
        if (entry.type === "newline") continue;
        wordIdx++;
        const l = entry.line || 1;
        if (!lines[l]) lines[l] = [];
        lines[l].push(entry.word || `[#${wordIdx}]`);
    }
    const lineNums = Object.keys(lines).map(Number).sort((a, b) => a - b);
    if (lineNums.length === 0) {
        $transcript.innerHTML = '<span class="transcript-empty">Record words to build transcript...</span>';
        return;
    }
    $transcript.innerHTML = lineNums.map(n => {
        const words = lines[n].map(w =>
            w.startsWith("[") ? `<span class="transcript-blank">${w}</span>` : w
        ).join(" ");
        return `<div class="transcript-line"><span class="transcript-ln">${n}</span>${words}</div>`;
    }).join("");
}
renderTranscript();

// ── Pending Queue ──────────────────────────────────────
function renderQueue() {
    const wordEntries = pendingQueue.filter(e => e.type === "word");
    $queueCount.textContent = wordEntries.length;
    $queueSection.classList.toggle("hidden", pendingQueue.length === 0);
    $queueList.innerHTML = "";
    let wordNum = 0;

    function addInsertBtn(insertAtIdx) {
        const wrap = document.createElement("div");
        wrap.className = "queue-insert-wrap";
        const btn = document.createElement("button");
        btn.className = "queue-insert-btn";
        btn.textContent = "+ New Line";
        btn.addEventListener("click", () => {
            const prevLine = insertAtIdx > 0 ? (pendingQueue[insertAtIdx - 1].line || currentLine) : 1;
            currentLine = prevLine + 1;
            pendingQueue.splice(insertAtIdx, 0, { type: "newline", line: currentLine });
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "mark_newline", line: currentLine }));
            }
            recalcLines();
            renderQueue();
            renderTranscript();
            $transcriptLineNum.textContent = `Line ${currentLine}`;
        });
        wrap.appendChild(btn);
        $queueList.appendChild(wrap);
    }

    for (let idx = 0; idx < pendingQueue.length; idx++) {
        const entry = pendingQueue[idx];

        if (entry.type === "newline") {
            const sep = document.createElement("div");
            sep.className = "queue-newline";

            const nlUp = document.createElement("button");
            nlUp.className = "btn-move"; nlUp.textContent = "\u25B2"; nlUp.title = "Move up";
            if (idx > 0) nlUp.addEventListener("click", () => moveQueueItem(idx, -1));
            else nlUp.disabled = true;
            sep.appendChild(nlUp);

            const nlDown = document.createElement("button");
            nlDown.className = "btn-move"; nlDown.textContent = "\u25BC"; nlDown.title = "Move down";
            if (idx < pendingQueue.length - 1) nlDown.addEventListener("click", () => moveQueueItem(idx, 1));
            else nlDown.disabled = true;
            sep.appendChild(nlDown);

            const nlLabel = document.createElement("span");
            nlLabel.className = "queue-newline-label";
            nlLabel.textContent = `--- Line ${entry.line} ---`;
            sep.appendChild(nlLabel);

            const removeBtn = document.createElement("button");
            removeBtn.className = "btn-secondary btn-xs"; removeBtn.textContent = "X";
            removeBtn.addEventListener("click", () => {
                pendingQueue.splice(idx, 1);
                recalcLines(); renderQueue(); renderTranscript();
            });
            sep.appendChild(removeBtn);
            $queueList.appendChild(sep);

            if (idx < pendingQueue.length - 1 && pendingQueue[idx + 1].type !== "newline") {
                addInsertBtn(idx + 1);
            }
            continue;
        }

        wordNum++;
        const card = document.createElement("div");
        card.className = "queue-card";

        const canvas = document.createElement("canvas");
        canvas.className = "queue-canvas";
        card.appendChild(canvas);

        const row = document.createElement("div");
        row.className = "queue-row";

        // Move arrows
        const arrows = document.createElement("div");
        arrows.className = "queue-arrows";
        const upBtn = document.createElement("button");
        upBtn.className = "btn-move"; upBtn.textContent = "\u25B2"; upBtn.title = "Move up";
        if (idx > 0) upBtn.addEventListener("click", () => moveQueueItem(idx, -1));
        else upBtn.disabled = true;
        arrows.appendChild(upBtn);
        const downBtn = document.createElement("button");
        downBtn.className = "btn-move"; downBtn.textContent = "\u25BC"; downBtn.title = "Move down";
        if (idx < pendingQueue.length - 1) downBtn.addEventListener("click", () => moveQueueItem(idx, 1));
        else downBtn.disabled = true;
        arrows.appendChild(downBtn);
        row.appendChild(arrows);

        const numLabel = document.createElement("span");
        numLabel.className = "queue-num"; numLabel.textContent = `#${wordNum}`;
        row.appendChild(numLabel);

        const wordInput = document.createElement("input");
        wordInput.type = "text"; wordInput.className = "queue-word";
        wordInput.value = entry.word; wordInput.placeholder = "type word...";
        wordInput.autocomplete = "off"; wordInput.spellcheck = false;
        wordInput.addEventListener("input", () => {
            entry.word = wordInput.value.trim().toLowerCase();
            renderTranscript();
        });
        row.appendChild(wordInput);

        const lineTag = document.createElement("span");
        lineTag.className = "queue-line-tag"; lineTag.textContent = `L${entry.line}`;
        row.appendChild(lineTag);

        const info = document.createElement("span");
        info.className = "queue-info"; info.textContent = `${entry.duration_s.toFixed(1)}s`;
        row.appendChild(info);

        if (entry.wavBlob) {
            const playBtn = document.createElement("button");
            playBtn.className = "btn-play"; playBtn.textContent = "\u25B6"; playBtn.title = "Play audio";
            playBtn.addEventListener("click", () => playAudio(entry.wavBlob, playBtn));
            row.appendChild(playBtn);
        }

        const saveBtn = document.createElement("button");
        saveBtn.className = "btn-primary btn-sm"; saveBtn.textContent = "Save";
        saveBtn.addEventListener("click", () => saveQueueItem(idx));
        row.appendChild(saveBtn);

        const discardBtn = document.createElement("button");
        discardBtn.className = "btn-secondary btn-sm"; discardBtn.textContent = "X"; discardBtn.title = "Discard";
        discardBtn.addEventListener("click", () => discardQueueItem(idx));
        row.appendChild(discardBtn);

        card.appendChild(row);
        $queueList.appendChild(card);

        if (idx < pendingQueue.length - 1 && pendingQueue[idx + 1].type !== "newline") {
            addInsertBtn(idx + 1);
        }

        requestAnimationFrame(() => {
            const miniChart = new AccelChart(canvas, {
                windowSec: Math.max(entry.duration_s * 1.1, 0.5), autoScale: true,
            });
            miniChart.setData(samplesToDeltas(entry.samples, entry.timestamps));
            miniChart.draw();
        });
    }
}

function recalcLines() {
    let num = 1;
    for (const entry of pendingQueue) {
        if (entry.type === "newline") { num++; entry.line = num; }
    }
    let cur = 1;
    for (const entry of pendingQueue) {
        if (entry.type === "newline") cur = entry.line;
        else entry.line = cur;
    }
    currentLine = cur;
    $transcriptLineNum.textContent = `Line ${currentLine}`;
}

function moveQueueItem(idx, direction) {
    const newIdx = idx + direction;
    if (newIdx < 0 || newIdx >= pendingQueue.length) return;
    const item = pendingQueue.splice(idx, 1)[0];
    pendingQueue.splice(newIdx, 0, item);
    recalcLines(); renderQueue(); renderTranscript();
}

let currentAudio = null;
function playAudio(wavBlob, btn) {
    if (currentAudio) {
        currentAudio.pause(); currentAudio = null;
        document.querySelectorAll(".btn-play.playing").forEach(b => {
            b.textContent = "\u25B6"; b.classList.remove("playing");
        });
    }
    const url = URL.createObjectURL(wavBlob);
    const audio = new Audio(url);
    currentAudio = audio;
    btn.textContent = "\u25A0"; btn.classList.add("playing");
    audio.onended = () => {
        btn.textContent = "\u25B6"; btn.classList.remove("playing");
        URL.revokeObjectURL(url); currentAudio = null;
    };
    audio.play();
}

async function uploadAudio(sampleId, wavBlob) {
    if (!wavBlob) return;
    try {
        await fetch(`/api/samples/${sampleId}/audio`, {
            method: "POST", headers: { "Content-Type": "audio/wav" }, body: wavBlob,
        });
    } catch (err) { console.warn("Audio upload failed:", err); }
}

async function saveQueueItem(idx) {
    const entry = pendingQueue[idx];
    if (entry.type === "newline") {
        pendingQueue.splice(idx, 1);
        recalcLines(); renderQueue(); renderTranscript();
        return;
    }
    const word = entry.word.trim().toLowerCase();
    if (!word) {
        const cards = $queueList.querySelectorAll(".queue-card");
        let wordCardIdx = 0;
        for (let i = 0; i < idx; i++) { if (pendingQueue[i].type === "word") wordCardIdx++; }
        const input = cards[wordCardIdx]?.querySelector(".queue-word");
        if (input) {
            input.style.borderColor = "#f85149"; input.focus();
            setTimeout(() => { input.style.borderColor = ""; }, 1000);
        }
        return;
    }

    await uploadAudio(entry.sample_id, entry.wavBlob);
    ws.send(JSON.stringify({
        type: "save_sample", sample_id: entry.sample_id, word, line: entry.line,
    }));

    savedHistory.unshift({
        id: entry.sample_id, word, line: entry.line,
        samples: entry.samples, timestamps: entry.timestamps,
        duration_s: entry.duration_s, num_samples: entry.num_samples,
    });
    pendingQueue.splice(idx, 1);
    renderQueue(); renderHistory(); renderTranscript();
}

function discardQueueItem(idx) {
    const entry = pendingQueue[idx];
    if (entry.type === "word") {
        ws.send(JSON.stringify({ type: "discard_sample", sample_id: entry.sample_id }));
    }
    pendingQueue.splice(idx, 1);
    recalcLines(); renderQueue(); renderTranscript();
}

async function saveAllQueue() {
    for (let i = pendingQueue.length - 1; i >= 0; i--) {
        const entry = pendingQueue[i];
        if (entry.type === "newline") { pendingQueue.splice(i, 1); continue; }
        const word = entry.word.trim().toLowerCase();
        if (!word) continue;
        await uploadAudio(entry.sample_id, entry.wavBlob);
        ws.send(JSON.stringify({
            type: "save_sample", sample_id: entry.sample_id, word, line: entry.line,
        }));
        savedHistory.unshift({
            id: entry.sample_id, word, line: entry.line,
            samples: entry.samples, timestamps: entry.timestamps,
            duration_s: entry.duration_s, num_samples: entry.num_samples,
        });
        pendingQueue.splice(i, 1);
    }
    renderQueue(); renderHistory(); renderTranscript();
    finalizeSession();
}

function discardAllQueue() {
    for (const entry of pendingQueue) {
        if (entry.type === "word") {
            ws.send(JSON.stringify({ type: "discard_sample", sample_id: entry.sample_id }));
        }
    }
    pendingQueue.length = 0;
    renderQueue();
}

$btnSaveAll.addEventListener("click", saveAllQueue);
$btnDiscardAll.addEventListener("click", discardAllQueue);

// ── Saved History ──────────────────────────────────────
function renderHistory() {
    $historyList.innerHTML = "";
    for (let idx = 0; idx < savedHistory.length; idx++) {
        const entry = savedHistory[idx];
        const card = document.createElement("div");
        card.className = "history-card";

        const header = document.createElement("div");
        header.className = "history-header";
        const wordEl = document.createElement("div");
        wordEl.className = "history-word"; wordEl.textContent = entry.word;
        header.appendChild(wordEl);
        const delBtn = document.createElement("button");
        delBtn.className = "history-delete"; delBtn.textContent = "Delete";
        delBtn.addEventListener("click", () => deleteSample(entry.id, idx));
        header.appendChild(delBtn);
        card.appendChild(header);

        const canvas = document.createElement("canvas");
        canvas.className = "history-canvas";
        canvas.style.width = "100%"; canvas.style.height = "80px";
        card.appendChild(canvas);

        const info = document.createElement("div");
        info.className = "history-info";
        info.textContent = `${entry.duration_s.toFixed(2)}s \u00b7 ${entry.num_samples} pts`;
        card.appendChild(info);

        $historyList.appendChild(card);

        requestAnimationFrame(() => {
            const miniChart = new AccelChart(canvas, {
                windowSec: Math.max(entry.duration_s * 1.1, 0.5), autoScale: true,
            });
            miniChart.setData(samplesToDeltas(entry.samples, entry.timestamps));
            miniChart.draw();
        });
    }
}

function deleteSample(sampleId, idx) {
    fetch(`/api/samples/${sampleId}`, { method: "DELETE" }).then(r => {
        if (r.ok) {
            savedHistory.splice(idx, 1);
            renderHistory();
            ws.send(JSON.stringify({ type: "get_stats" }));
        }
    });
}

// ── Dataset UI ─────────────────────────────────────────
function updateDatasetUI(data) {
    $totalSamples.textContent = data.total_samples + " samples";
    const words = data.words || {};
    const sorted = Object.entries(words).sort((a, b) => b[1] - a[1]);
    $wordCounts.innerHTML = sorted.map(([w, c]) =>
        `<span class="word-chip">${w} <span class="count">${c}</span></span>`
    ).join("");
    $datasetTotal.textContent = data.total_samples > 0
        ? `${data.total_samples} total samples, ${data.total_duration_s || 0}s of data`
        : "No samples yet \u2014 start recording!";
}
