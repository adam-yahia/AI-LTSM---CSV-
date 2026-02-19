/**
 * functions.js
 * ─────────────────────────────────────────────────────────
 * All core logic for MedPredict:
 *  - Configuration initialization (min/max bounds)
 *  - Data normalization / denormalization
 *  - Input vector construction (8 features, no neighbourhood)
 *  - Training data preparation with minority oversampling
 *  - Model training with live progress callbacks
 *  - Accuracy evaluation
 *  - Prediction
 *  - DOM helpers (architecture viz, table, neighbourhood select)
 *
 * WHY NO NEIGHBOURHOOD ENCODING:
 *   37 unique neighbourhoods × one-hot = 37 extra near-zero
 *   features on top of only 150 samples. The network cannot
 *   learn meaningful weights for rare categories and collapses
 *   to predicting the majority class (show=0) for everything.
 *   Removing it gives the network 8 clean, informative features.
 *
 * WHY OVERSAMPLING:
 *   The dataset is 78% "showed up" / 22% "no-show". Without
 *   correction the network minimizes loss by always guessing
 *   "showed up". We duplicate no-show records 3× to balance.
 * ─────────────────────────────────────────────────────────
 */

import { RAW_DATA, configuration, myBrain, toggleState } from './config.js';

/* ══════════════════════════════════════════════════════════
   CONFIGURATION INIT
══════════════════════════════════════════════════════════ */
export function initConfiguration() {
  RAW_DATA.forEach(r => {
    if (r.age       < configuration.age.min)  configuration.age.min  = r.age;
    if (r.age       > configuration.age.max)  configuration.age.max  = r.age;
    if (r.days_wait < configuration.days.min) configuration.days.min = r.days_wait;
    if (r.days_wait > configuration.days.max) configuration.days.max = r.days_wait;
    if (!configuration.hoods.includes(r.neighbourhood))
      configuration.hoods.push(r.neighbourhood);
  });
  configuration.hoods.sort();
}

/* ══════════════════════════════════════════════════════════
   NORMALIZATION
══════════════════════════════════════════════════════════ */
export function normalizeData(value, min, max) {
  if (min === max) return 0.5;
  return (value - min) / (max - min);
}

export function denormalizeData(value, min, max) {
  return value * (max - min) + min;
}

/* ══════════════════════════════════════════════════════════
   INPUT VECTOR BUILDER  (8 features, no neighbourhood)
══════════════════════════════════════════════════════════ */
export function buildInputVector(age, days, gender, sms, schl, ht, db, al) {
  return {
    age:          normalizeData(age,  configuration.age.min,  configuration.age.max),
    days_wait:    normalizeData(days, configuration.days.min, configuration.days.max),
    gender:       gender,
    sms_received: sms,
    scholarship:  schl,
    hipertension: ht,
    diabetes:     db,
    alcoholism:   al
  };
}

/* ══════════════════════════════════════════════════════════
   PREPARE TRAINING DATA  (with minority-class oversampling)
   No-show records are repeated 3× so the network sees a
   roughly balanced class distribution (~50/50) and learns
   to distinguish both outcomes instead of always predicting
   the majority class.
══════════════════════════════════════════════════════════ */
export function prepareTrainingData() {
  const OVERSAMPLE_FACTOR = 3;   // repeat no-show rows this many times
  const samples = [];

  RAW_DATA.forEach(r => {
    const entry = {
      input: buildInputVector(
        r.age, r.days_wait, r.gender, r.sms_received,
        r.scholarship, r.hipertension, r.diabetes, r.alcoholism
      ),
      output: { noshow: r.noshow }
    };
    samples.push(entry);
    // Oversample minority class
    if (r.noshow === 1) {
      for (let i = 1; i < OVERSAMPLE_FACTOR; i++) {
        samples.push(entry);
      }
    }
  });

  // Shuffle so no-show/show records aren't grouped together
  for (let i = samples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [samples[i], samples[j]] = [samples[j], samples[i]];
  }

  myBrain.trainingData = samples;
  return samples;
}

/* ══════════════════════════════════════════════════════════
   TRAIN MODEL
══════════════════════════════════════════════════════════ */
export function trainModel({ onLog, onProgress, onDone, onError }) {
  const trainingData = prepareTrainingData();
  const ITERATIONS   = 5000;
  const LOG_PERIOD   = 250;

  onLog(`[INIT] Original samples: 150 → After oversampling: ${trainingData.length}`);
  onLog(`[INIT] Input features: 8 (age, days_wait, gender, sms, scholarship, hypertension, diabetes, alcoholism)`);
  onLog(`[INIT] Architecture: 8 → [10, 6] → 1 (sigmoid)`);
  onLog(`[INIT] Learning rate: 0.01 · Max iterations: ${ITERATIONS}`);
  onLog('─'.repeat(52));

  setTimeout(() => {
    try {
      const stats = myBrain.net.train(trainingData, {
        iterations:     ITERATIONS,
        errorThresh:    0.01,
        learningRate:   0.01,
        log:            true,
        logPeriod:      LOG_PERIOD,
        callback: (info) => {
          const pct = Math.min(100, Math.round((info.iterations / ITERATIONS) * 100));
          onProgress({ pct, error: info.error });
          onLog(`[iter ${String(info.iterations).padStart(4, '0')}]  error: ${info.error.toFixed(6)}`);
        },
        callbackPeriod: LOG_PERIOD
      });

      onLog('─'.repeat(52));
      onLog(`[DONE] Finished in ${stats.iterations} iterations · Final error: ${stats.error.toFixed(6)}`);
      onDone(stats);
    } catch (err) {
      onError(err);
    }
  }, 50);
}

/* ══════════════════════════════════════════════════════════
   EVALUATE ACCURACY  (on original 150 rows, not oversampled)
══════════════════════════════════════════════════════════ */
export function evaluateAccuracy() {
  const THRESHOLD = 0.5;
  let correct = 0, tpShow = 0, fnShow = 0, tpNoshow = 0, fnNoshow = 0;

  RAW_DATA.forEach(r => {
    const inp = buildInputVector(
      r.age, r.days_wait, r.gender, r.sms_received,
      r.scholarship, r.hipertension, r.diabetes, r.alcoholism
    );
    const raw     = myBrain.net.run(inp);
    const predVal = typeof raw === 'object' && !Array.isArray(raw)
      ? raw.noshow
      : (Array.isArray(raw) ? raw[0] : raw);
    const predClass = predVal > THRESHOLD ? 1 : 0;
    const actual    = r.noshow;

    if (predClass === actual) {
      correct++;
      if (actual === 0) tpShow++;
      else              tpNoshow++;
    } else {
      if (actual === 0) fnShow++;
      else              fnNoshow++;
    }
  });

  return {
    overall:      ((correct / RAW_DATA.length) * 100).toFixed(1),
    showUpRecall: tpShow + fnShow > 0
      ? ((tpShow    / (tpShow    + fnShow))    * 100).toFixed(1) : '0.0',
    noshowRecall: tpNoshow + fnNoshow > 0
      ? ((tpNoshow  / (tpNoshow  + fnNoshow))  * 100).toFixed(1) : '0.0'
  };
}

/* ══════════════════════════════════════════════════════════
   PREDICT
══════════════════════════════════════════════════════════ */
export function predict() {
  const age  = parseInt(document.getElementById('inpAge').value)  || 30;
  const days = parseInt(document.getElementById('inpDays').value) || 0;

  const gender = toggleState.gender === 'M' ? 1 : 0;
  const sms    = toggleState.sms    === 'Yes' ? 1 : 0;
  const schl   = toggleState.schl   === 'Yes' ? 1 : 0;
  const ht     = toggleState.ht     === 'Yes' ? 1 : 0;
  const db     = toggleState.db     === 'Yes' ? 1 : 0;
  const al     = toggleState.al     === 'Yes' ? 1 : 0;

  const input  = buildInputVector(age, days, gender, sms, schl, ht, db, al);
  const raw    = myBrain.net.run(input);
  return typeof raw === 'object' && !Array.isArray(raw)
    ? raw.noshow
    : (Array.isArray(raw) ? raw[0] : raw);
}

/* ══════════════════════════════════════════════════════════
   DOM HELPERS
══════════════════════════════════════════════════════════ */
export function buildArchViz() {
  const wrap = document.getElementById('archViz');
  const layers = [
    { label: 'Input (8)',    count: 8, type: 'input'  },
    { label: 'Hidden 1 (10)', count: 9, type: 'hidden' },
    { label: 'Hidden 2 (6)', count: 6, type: 'hidden' },
    { label: 'Output (1)',   count: 1, type: 'output' }
  ];

  layers.forEach((layer, li) => {
    if (li > 0) {
      const arrow = document.createElement('div');
      arrow.textContent = '→';
      arrow.style.cssText = 'color:#1a3050;font-size:18px;margin-top:40px;';
      wrap.appendChild(arrow);
    }
    const col   = document.createElement('div');
    col.className = 'arch-layer';
    const label = document.createElement('div');
    label.className = 'arch-layer-label';
    label.textContent = layer.label;
    const nodes = document.createElement('div');
    nodes.className = 'arch-nodes';
    for (let i = 0; i < layer.count; i++) {
      const node = document.createElement('div');
      node.className = 'arch-node' + (layer.type === 'output' ? ' output-node' : '');
      nodes.appendChild(node);
    }
    col.appendChild(label);
    col.appendChild(nodes);
    wrap.appendChild(col);
  });
}

export function activateArchNodes() {
  document.querySelectorAll('.arch-node').forEach(n => n.classList.add('active'));
}

export function populateNeighbourhoodSelect() {
  const sel = document.getElementById('inpNeighbourhood');
  configuration.hoods.forEach(h => {
    const opt = document.createElement('option');
    opt.value = h;
    opt.textContent = h;
    sel.appendChild(opt);
  });
}

export function populateDataTable() {
  const tbody = document.getElementById('dataTableBody');
  RAW_DATA.slice(0, 10).forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:var(--muted)">${i + 1}</td>
      <td>${r.age}</td>
      <td>${r.gender === 1 ? 'M' : 'F'}</td>
      <td>${r.days_wait}</td>
      <td>${r.sms_received  ? '✓' : '-'}</td>
      <td>${r.scholarship   ? '✓' : '-'}</td>
      <td>${r.hipertension  ? '✓' : '-'}</td>
      <td>${r.diabetes      ? '✓' : '-'}</td>
      <td>${r.alcoholism    ? '✓' : '-'}</td>
      <td style="max-width:140px;overflow:hidden;text-overflow:ellipsis">${r.neighbourhood}</td>
      <td><span class="pill ${r.noshow ? 'pill-yes' : 'pill-no'}">${r.noshow ? 'YES' : 'NO'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

export function updateGauge(risk) {
  const CIRCUMFERENCE = 301.6;
  const pct           = Math.round(risk * 100);
  const isNoShow      = risk > 0.5;
  const color         = isNoShow ? 'var(--danger)' : 'var(--accent)';

  const circle = document.getElementById('gaugeCircle');
  circle.style.strokeDashoffset = CIRCUMFERENCE - risk * CIRCUMFERENCE;
  circle.style.stroke           = color;
  circle.style.transition       = 'stroke-dashoffset 0.8s ease, stroke 0.5s';
  document.getElementById('gaugeText').textContent = pct + '%';

  const lbl  = document.getElementById('verdictLabel');
  const desc = document.getElementById('verdictDesc');

  if (isNoShow) {
    lbl.className    = 'verdict-label no-show';
    lbl.textContent  = 'Likely No-Show';
    desc.textContent = `Risk score: ${pct}% — High probability of missing the appointment. Consider a manual follow-up or confirmation call.`;
  } else {
    lbl.className    = 'verdict-label will-show';
    lbl.textContent  = 'Will Attend';
    desc.textContent = `Risk score: ${pct}% — Patient is likely to attend. No special intervention required.`;
  }
}
