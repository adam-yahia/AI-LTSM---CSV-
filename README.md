# AI AppointmentsPro

A browser-based demo with two AI models:

1) **CSV No-Show Predictor (Neural Network)**  
   Train a model from a CSV dataset and evaluate accuracy (test set).  
2) **Text LSTM**  
   Train an LSTM on a text file and generate short text outputs.

## How to run
- Open `web/index.html` in a browser  
  *(or use VS Code → Live Server for best results)*

## Files
- `web/index.html` — UI
- `web/app.js` — training / evaluation logic (CSV + LSTM)
- `data/` — sample datasets (CSV / TXT)

## Notes
Models train locally in the browser. Dataset size affects training time and results.
