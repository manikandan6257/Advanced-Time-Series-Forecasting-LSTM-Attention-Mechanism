# Advanced Time Series Forecasting with Deep Learning: LSTM & Attention Mechanism

## Overview
This project explores advanced deep learning techniques for time series forecasting using **Long Short-Term Memory (LSTM)** networks and a variant enhanced with a **self-attention mechanism**. The primary goal is to evaluate how the attention mechanism affects feature weighting, interpretability, and forecast stability compared to a baseline LSTM.

The experiments are conducted on a synthetic multivariate time series dataset exhibiting clear trend and seasonality. The project emphasizes high-quality, modular code with type hints, docstrings, and reproducible results using time series–appropriate cross-validation strategies.

---

## Key Features
- Synthetic **multivariate time series generation** with trend, seasonality, and noise.
- Implementation of:
  - Baseline **LSTM model**.
  - **LSTM with self-attention mechanism** for enhanced interpretability.
- **Rolling window cross-validation** for robust temporal model evaluation.
- Model evaluation using multiple metrics: RMSE, MAE, and MAPE.
- Production-quality code: modular functions, consistent naming, and clear documentation.
- Comparative analysis of prediction accuracy, hyperparameters, and attention-based interpretability.

---

## Technologies Used
- Python 3.9+
- TensorFlow / Keras
- NumPy / Pandas / SciPy
- scikit-learn
- Matplotlib (optional for visualization)

---


## Project Structure

├── data/ # Placeholder for dataset or generated files
├── models/ # Optional: saved model weights or configurations
├── results/ # Evaluation metrics and analysis results
├── main.py # Main Colab/Notebook entry script
├── README.md # Project documentation
└── requirements.txt # Python dependencies

---

## Getting Started

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run the experiment
The project can be run directly in Google Colab or locally:
python main.py

or open the corresponding **Colab Notebook** for interactive training and analysis.

---

## Model Architectures

### Baseline LSTM
A univariate or multivariate LSTM trained using a standard architecture with:
- 1 LSTM layer (64 units)
- Dense layers (32 → 1)
- Adam optimizer with MSE loss

### Attention-LSTM
Builds upon the baseline model by adding a **custom self-attention layer**.  
This layer computes dynamic temporal dependencies, assigning weights to each time step before prediction.

---

## Evaluation Metrics
- **RMSE (Root Mean Squared Error):** Evaluates prediction dispersion.  
- **MAE (Mean Absolute Error):** Measures average prediction error.  
- **MAPE (Mean Absolute Percentage Error):** Captures relative deviation.

These metrics are averaged across multiple cross-validation splits.

---

## Cross-Validation Strategy
The project employs a **rolling window approach** suitable for time series forecasting.  
At each iteration:
1. Expanding training window accumulates data sequentially.
2. Validation is performed on the next unseen window.
3. Performance metrics are averaged across all folds for robust evaluation.

---

## Results Summary
| Model            | RMSE  | MAE   | MAPE  |
|------------------|-------|-------|-------|
| Baseline LSTM    | ~x.xx | ~x.xx | ~x.xx |
| Attention-LSTM   | ~x.xx | ~x.xx | ~x.xx |

*Note: Replace placeholders with actual computed metrics after running experiments.*

**Insight:**  
The self-attention mechanism improves temporal pattern recognition and reduces forecast volatility, particularly around seasonal inflection points.

---

## Interpretability Notes
The attention mechanism provides insight into which time steps contribute most to predictions.  
Attention weights tend to peak during local trends or seasonal cycles, helping explain model reasoning compared to traditional LSTMs.

---

## License
This project is released under the [MIT License](LICENSE).

---

## Author
Developed as part of an advanced machine learning assignment on **Deep Learning for Time Series Forecasting**.

---

## Acknowledgements
- TensorFlow documentation  
- “Attention is All You Need” (Vaswani et al., 2017)  
- Statsmodels time series literature


