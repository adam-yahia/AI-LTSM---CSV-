/**
 * lstm.worker.js
 * ─────────────────────────────────────────────────────────
 * Runs inside a Web Worker so LSTM training never blocks
 * the main thread (which causes browser tab crashes/freezes).
 *
 * Communication via postMessage:
 *   Main → Worker:  { type: 'train', data: RAW_DATA }
 *   Worker → Main:  { type: 'log',      message }
 *                   { type: 'progress', pct, error }
 *                   { type: 'done',     metrics }
 *                   { type: 'error',    message }
 *   Main → Worker:  { type: 'predict',  text }
 *   Worker → Main:  { type: 'prediction', label, cleaned }
 * ─────────────────────────────────────────────────────────
 */

// Load brain.js inside the worker
importScripts('https://cdn.jsdelivr.net/npm/brain.js@1.6.0/browser.js');

let net = null;

/* ══════════════════════════════════════════════════════════
   TEXT HELPERS
══════════════════════════════════════════════════════════ */
function recordToText(r) {
  const gender   = r.gender === 1 ? 'male' : 'female';
  const days     = r.days_wait === 0 ? 'sameday' : `wait${r.days_wait}`;
  const sms      = r.sms_received ? 'sms' : 'nosms';
  const schl     = r.scholarship  ? 'scholarship' : '';
  const hypert   = r.hipertension ? 'hypertension' : '';
  const diabetes = r.diabetes     ? 'diabetes' : '';
  const alcohol  = r.alcoholism   ? 'alcoholism' : '';
  // Keep texts short — LSTM trains faster on short sequences
  return [gender, `age${r.age}`, days, sms, schl, hypert, diabetes, alcohol]
    .filter(Boolean).join(' ');
}

function cleanText(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, ' ').trim();
}

/* ══════════════════════════════════════════════════════════
   DATASET BUILDER
══════════════════════════════════════════════════════════ */
function buildDataset(RAW_DATA) {
  const all = RAW_DATA.map(r => ({
    input:  cleanText(recordToText(r)),
    output: r.noshow === 1 ? 'no' : 'yes',  // short labels = faster LSTM
    noshow: r.noshow
  }));

  // Shuffle
  for (let i = all.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [all[i], all[j]] = [all[j], all[i]];
  }

  const trainEnd = Math.floor(all.length * 0.70);
  const valEnd   = Math.floor(all.length * 0.85);

  const rawTrain   = all.slice(0, trainEnd);
  const validation = all.slice(trainEnd, valEnd);
  const test       = all.slice(valEnd);

  // Oversample no-show 3×
  const train = [...rawTrain];
  rawTrain.forEach(s => {
    if (s.noshow === 1) { train.push(s); train.push(s); }
  });

  // Shuffle again
  for (let i = train.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [train[i], train[j]] = [train[j], train[i]];
  }

  return { train, validation, test };
}

/* ══════════════════════════════════════════════════════════
   EVALUATE
══════════════════════════════════════════════════════════ */
function evalSet(set) {
  let correct = 0, tpShow = 0, fnShow = 0, tpNoshow = 0, fnNoshow = 0;
  set.forEach(s => {
    const out       = net.run(s.input) || '';
    const predLabel = out.toLowerCase().trim().startsWith('n') ? 'no' : 'yes';
    const isCorrect = predLabel === s.output;
    if (isCorrect) {
      correct++;
      if (s.output === 'yes') tpShow++;
      else                    tpNoshow++;
    } else {
      if (s.output === 'yes') fnShow++;
      else                    fnNoshow++;
    }
  });
  return {
    acc:          set.length > 0 ? ((correct / set.length) * 100).toFixed(1) : '0.0',
    showRecall:   tpShow    + fnShow    > 0 ? ((tpShow    / (tpShow    + fnShow))    * 100).toFixed(1) : '0.0',
    noshowRecall: tpNoshow  + fnNoshow  > 0 ? ((tpNoshow  / (tpNoshow  + fnNoshow))  * 100).toFixed(1) : '0.0'
  };
}

/* ══════════════════════════════════════════════════════════
   MESSAGE HANDLER
══════════════════════════════════════════════════════════ */
self.onmessage = function(e) {
  const { type, data, text } = e.data;

  if (type === 'train') {
    try {
      const { train, validation, test } = buildDataset(data);

      const examples = train.slice(0, 3)
        .map(s => `"${s.input}" → ${s.output}`).join('\n');

      postMessage({ type: 'samples', examples });
      postMessage({ type: 'log', message: `[INIT] Train: ${train.length} · Val: ${validation.length} · Test: ${test.length}` });
      postMessage({ type: 'log', message: `[INIT] Short labels: "yes" (show) / "no" (no-show)` });
      postMessage({ type: 'log', message: `[INIT] Iterations: 300 · LR: 0.01` });
      postMessage({ type: 'log', message: '─'.repeat(48) });

      net = new brain.recurrent.LSTM();

      const ITERATIONS  = 300;
      const LOG_PERIOD  = 30;
      let lastLogIter   = 0;

      const trainingInput = train.map(s => ({ input: s.input, output: s.output }));

      net.train(trainingInput, {
        iterations:     ITERATIONS,
        errorThresh:    0.01,
        learningRate:   0.01,
        log:            true,
        logPeriod:      LOG_PERIOD,
        callback: (info) => {
          const pct = Math.min(100, Math.round((info.iterations / ITERATIONS) * 100));
          postMessage({ type: 'progress', pct, error: info.error });
          postMessage({ type: 'log', message: `[iter ${String(info.iterations).padStart(4,'0')}]  error: ${info.error.toFixed(6)}` });
          lastLogIter = info.iterations;
        },
        callbackPeriod: LOG_PERIOD
      });

      postMessage({ type: 'log', message: '─'.repeat(48) });
      postMessage({ type: 'log', message: '[DONE] LSTM training complete.' });

      const valM  = evalSet(validation);
      const testM = evalSet(test);

      postMessage({ type: 'log', message: `[EVAL] Val accuracy:       ${valM.acc}%` });
      postMessage({ type: 'log', message: `[EVAL] Test accuracy:      ${testM.acc}%` });
      postMessage({ type: 'log', message: `[EVAL] Test no-show recall: ${testM.noshowRecall}%` });

      postMessage({
        type: 'done',
        metrics: {
          valAcc:           valM.acc,
          testAcc:          testM.acc,
          testShowRecall:   testM.showRecall,
          testNoshowRecall: testM.noshowRecall
        }
      });

    } catch (err) {
      postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'predict') {
    try {
      const cleaned = cleanText(text);
      const out     = net.run(cleaned) || '';
      const label   = out.toLowerCase().trim().startsWith('n') ? 'noshow' : 'showup';
      postMessage({ type: 'prediction', label, cleaned });
    } catch (err) {
      postMessage({ type: 'error', message: err.message });
    }
  }
};
