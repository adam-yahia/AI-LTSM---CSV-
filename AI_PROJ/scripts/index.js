/**
 * index.js
 * ─────────────────────────────────────────────────────────
 * Application entry point.
 * Wires all DOM events and orchestrates the UI flow:
 *   1. Init on DOMContentLoaded
 *   2. Neural network train button → training pipeline
 *   3. Toggle buttons → update toggleState
 *   4. Neural network predict form → inference + result
 *   5. LSTM train button → Web Worker training pipeline
 *   6. LSTM predict button → async Worker inference + result
 * ─────────────────────────────────────────────────────────
 */

import { toggleState, RAW_DATA } from './config.js';
import {
  initConfiguration,
  buildArchViz,
  populateDataTable,
  activateArchNodes,
  trainModel,
  evaluateAccuracy,
  predict,
  updateGauge
} from './functions.js';

import {
  trainLstm,
  predictFromText,
  recordToText,
  cleanText,
  lstmState
} from './lstm.js';

/* ══════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════ */
window.addEventListener('DOMContentLoaded', () => {
  initConfiguration();
  buildArchViz();
  populateDataTable();
  wireToggleButtons();
  wireTrainButton();
  wirePredictForm();
  wireLstmTrainButton();
  wireLstmPredictButton();
});

/* ══════════════════════════════════════════════════════════
   TOGGLE BUTTONS
══════════════════════════════════════════════════════════ */
function wireToggleButtons() {
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.toggle;
      const val = btn.dataset.val;
      toggleState[key] = val;
      document.querySelectorAll(`.toggle-btn[data-toggle="${key}"]`)
        .forEach(s => s.classList.toggle('selected', s === btn));
    });
  });
}

/* ══════════════════════════════════════════════════════════
   NEURAL NETWORK — TRAIN
══════════════════════════════════════════════════════════ */
function wireTrainButton() {
  document.getElementById('trainBtn').addEventListener('click', startTraining);
}

function startTraining() {
  const btn        = document.getElementById('trainBtn');
  const progressEl = document.getElementById('progressWrap');
  const fillEl     = document.getElementById('progressFill');
  const pctEl      = document.getElementById('progressPct');
  const statusEl   = document.getElementById('progressStatus');
  const logEl      = document.getElementById('logOutput');

  btn.disabled       = true;
  btn.textContent    = '⏳ Training…';
  progressEl.classList.add('visible');
  logEl.innerHTML    = '';
  fillEl.style.width = '0%';

  activateArchNodes();

  trainModel({
    onLog: msg => { logEl.innerHTML += msg + '\n'; logEl.scrollTop = logEl.scrollHeight; },
    onProgress: ({ pct, error }) => {
      fillEl.style.width   = pct + '%';
      pctEl.textContent    = pct + '%';
      statusEl.textContent = `Training… (error: ${error.toFixed(4)})`;
    },
    onDone: () => {
      fillEl.style.width   = '100%';
      pctEl.textContent    = '100%';
      statusEl.textContent = '✓ Training Complete';
      btn.textContent      = '✓ Trained';
      btn.style.background = 'var(--accent)';

      const acc = evaluateAccuracy();
      document.getElementById('accOverall').textContent = acc.overall + '%';
      document.getElementById('accShowUp').textContent  = acc.showUpRecall + '%';
      document.getElementById('accNoshow').textContent  = acc.noshowRecall + '%';
      document.getElementById('accuracySection').style.display = 'block';

      document.getElementById('predictCard').classList.add('enabled');
      document.getElementById('activeDot').style.display = 'inline-block';
    },
    onError: err => {
      logEl.innerHTML += `[ERROR] ${err.message}\n`;
      btn.disabled    = false;
      btn.textContent = '▶ Retry Training';
    }
  });
}

/* ══════════════════════════════════════════════════════════
   NEURAL NETWORK — PREDICT
══════════════════════════════════════════════════════════ */
function wirePredictForm() {
  document.getElementById('predictForm').addEventListener('submit', e => {
    e.preventDefault();
    if (!document.getElementById('predictCard').classList.contains('enabled')) return;
    const risk = predict();
    document.getElementById('resultPanel').classList.add('visible');
    updateGauge(risk);
  });
}

/* ══════════════════════════════════════════════════════════
   LSTM — TRAIN  (via Web Worker — never blocks main thread)
══════════════════════════════════════════════════════════ */
function wireLstmTrainButton() {
  document.getElementById('lstmTrainBtn').addEventListener('click', startLstmTraining);
}

function startLstmTraining() {
  const btn        = document.getElementById('lstmTrainBtn');
  const progressEl = document.getElementById('lstmProgressWrap');
  const fillEl     = document.getElementById('lstmProgressFill');
  const pctEl      = document.getElementById('lstmProgressPct');
  const statusEl   = document.getElementById('lstmProgressStatus');
  const logEl      = document.getElementById('lstmLogOutput');
  const sampleEl   = document.getElementById('lstmSampleText');

  btn.disabled       = true;
  btn.textContent    = '⏳ Training LSTM…';
  progressEl.classList.add('visible');
  logEl.innerHTML    = '';
  fillEl.style.width = '0%';
  sampleEl.textContent = 'Generating training texts…';

  trainLstm({
    onSamples: examples => {
      sampleEl.textContent = examples;
    },
    onLog: msg => {
      logEl.innerHTML += msg + '\n';
      logEl.scrollTop  = logEl.scrollHeight;
    },
    onProgress: ({ pct, error }) => {
      fillEl.style.width   = pct + '%';
      pctEl.textContent    = pct + '%';
      statusEl.textContent = `Training LSTM… (error: ${error.toFixed(4)})`;
    },
    onDone: metrics => {
      fillEl.style.width   = '100%';
      pctEl.textContent    = '100%';
      statusEl.textContent = '✓ LSTM Training Complete';
      btn.textContent      = '✓ LSTM Trained';
      btn.style.background = 'var(--warn)';

      document.getElementById('lstmValAcc').textContent       = metrics.valAcc + '%';
      document.getElementById('lstmTestAcc').textContent      = metrics.testAcc + '%';
      document.getElementById('lstmShowRecall').textContent   = metrics.testShowRecall + '%';
      document.getElementById('lstmNoshowRecall').textContent = metrics.testNoshowRecall + '%';
      document.getElementById('lstmAccuracySection').style.display = 'block';

      document.getElementById('lstmPredictCard').classList.add('enabled');
      document.getElementById('lstmActiveDot').style.display = 'inline-block';
    },
    onError: err => {
      logEl.innerHTML += `[ERROR] ${err.message}\n`;
      btn.disabled    = false;
      btn.textContent = '▶ Retry LSTM Training';
    }
  });
}

/* ══════════════════════════════════════════════════════════
   LSTM — PREDICT  (async — result comes back from worker)
══════════════════════════════════════════════════════════ */
function wireLstmPredictButton() {
  document.getElementById('lstmPredictBtn').addEventListener('click', runLstmPrediction);

  // Live-preview cleaned text while user types
  document.getElementById('lstmTextInput').addEventListener('input', e => {
    const cleaned = cleanText(e.target.value);
    document.getElementById('lstmCleanedText').textContent =
      cleaned ? `Cleaned: "${cleaned}"` : '';
  });
}

async function runLstmPrediction() {
  const card = document.getElementById('lstmPredictCard');
  if (!card.classList.contains('enabled')) return;

  const rawText = document.getElementById('lstmTextInput').value.trim();
  if (!rawText) return;

  const btn = document.getElementById('lstmPredictBtn');
  btn.textContent = '⏳ Predicting…';
  btn.disabled    = true;

  try {
    const { label, cleaned } = await predictFromText(rawText);
    const isNoShow = label === 'noshow';

    document.getElementById('lstmResultPanel').classList.add('visible');

    const circle = document.getElementById('lstmGaugeCircle');
    circle.style.strokeDashoffset = isNoShow ? 75 : 226;
    circle.style.stroke           = isNoShow ? 'var(--danger)' : 'var(--accent)';
    circle.style.transition       = 'stroke-dashoffset 0.8s ease, stroke 0.5s';
    document.getElementById('lstmGaugeText').textContent = isNoShow ? 'NO-SHOW' : 'SHOW';

    const lbl  = document.getElementById('lstmVerdictLabel');
    const desc = document.getElementById('lstmVerdictDesc');

    if (isNoShow) {
      lbl.className    = 'verdict-label no-show';
      lbl.textContent  = 'Likely No-Show';
      desc.textContent = `LSTM read: "${cleaned}" → predicted the patient will miss their appointment.`;
    } else {
      lbl.className    = 'verdict-label will-show';
      lbl.textContent  = 'Will Attend';
      desc.textContent = `LSTM read: "${cleaned}" → predicted the patient will attend.`;
    }
  } catch (err) {
    alert('Prediction error: ' + err.message);
  } finally {
    btn.textContent = '🔮 Predict with LSTM';
    btn.disabled    = false;
  }
}
