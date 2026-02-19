/**
 * lstm.js
 * ─────────────────────────────────────────────────────────
 * Manages the LSTM Web Worker.
 * All heavy training runs in lstm.worker.js (off main thread)
 * so the browser never freezes or crashes.
 *
 * PIPELINE (per assignment):
 *   1. Text generation  — recordToText()
 *   2. Text cleaning    — cleanText()
 *   3. Train/Val/Test split (70/15/15) + oversampling  [in worker]
 *   4. LSTM training via brain.recurrent.LSTM           [in worker]
 *   5. Evaluation on val + test sets                    [in worker]
 *   6. Free-text prediction                             [in worker]
 * ─────────────────────────────────────────────────────────
 */

import { RAW_DATA } from './config.js';

/* ── Worker singleton ─────────────────────────────────── */
let worker = null;

export const lstmState = { trained: false };

/* ══════════════════════════════════════════════════════════
   TEXT HELPERS  (mirrored here for the UI preview only)
══════════════════════════════════════════════════════════ */
export function recordToText(r) {
  const gender   = r.gender === 1 ? 'male' : 'female';
  const days     = r.days_wait === 0 ? 'sameday' : `wait${r.days_wait}`;
  const sms      = r.sms_received ? 'sms' : 'nosms';
  const schl     = r.scholarship  ? 'scholarship' : '';
  const hypert   = r.hipertension ? 'hypertension' : '';
  const diabetes = r.diabetes     ? 'diabetes' : '';
  const alcohol  = r.alcoholism   ? 'alcoholism' : '';
  return [gender, `age${r.age}`, days, sms, schl, hypert, diabetes, alcohol]
    .filter(Boolean).join(' ');
}

export function cleanText(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, ' ').trim();
}

/* ══════════════════════════════════════════════════════════
   TRAIN  — spins up the worker and streams callbacks back
══════════════════════════════════════════════════════════ */
export function trainLstm({ onLog, onProgress, onSamples, onDone, onError }) {
  // Terminate any previous worker
  if (worker) { worker.terminate(); worker = null; }

  worker = new Worker('./scripts/lstm.worker.js');

  worker.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case 'samples':   onSamples(msg.examples);                    break;
      case 'log':       onLog(msg.message);                         break;
      case 'progress':  onProgress({ pct: msg.pct, error: msg.error }); break;
      case 'done':
        lstmState.trained = true;
        onDone(msg.metrics);
        break;
      case 'error':     onError(new Error(msg.message));            break;
    }
  };

  worker.onerror = (err) => {
    onError(new Error(err.message || 'Worker error'));
  };

  // Send the dataset to the worker
  worker.postMessage({ type: 'train', data: RAW_DATA });
}

/* ══════════════════════════════════════════════════════════
   PREDICT  — sends text to worker, returns via Promise
══════════════════════════════════════════════════════════ */
export function predictFromText(rawText) {
  return new Promise((resolve, reject) => {
    if (!worker || !lstmState.trained) {
      return reject(new Error('LSTM not trained yet'));
    }

    // One-time listener for the prediction response
    const handler = (e) => {
      if (e.data.type === 'prediction' || e.data.type === 'error') {
        worker.removeEventListener('message', handler);
        if (e.data.type === 'error') reject(new Error(e.data.message));
        else resolve({ label: e.data.label, cleaned: e.data.cleaned });
      }
    };

    worker.addEventListener('message', handler);
    worker.postMessage({ type: 'predict', text: rawText });
  });
}
