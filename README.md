# Deep Learning Models on High-Frequency Trading Data 📈

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![ArXiv](https://img.shields.io/badge/Inspired%20By-Research%20Papers-orange.svg)]()

This repository explores **deep learning architectures** applied to **high-frequency limit order book (LOB)** and **event sequence data**, inspired by recent advances in deep learning for financial time-series modeling.

---

## 📘 Overview

This project investigates deep learning methods for **short-term price prediction** using **tick-by-tick market microstructure data** from the **National Stock Exchange (NSE)**.  
Our goal is to predict the **mid-price 10 seconds into the future** based on historical limit order book states and event sequences.

---

## 📊 Dataset Description

- **Exchange:** National Stock Exchange (NSE)  
- **Duration:** 66 trading days  
- **Frequency:** Tick-level (high-frequency)  
- **Samples:** ~0.6 million  

Each sample includes:

**Limit Order Book (LOB) features**
- Bid/Ask **price** and **size** for **10 levels**
- Last **100 historical timestamps** of the same stock

**Event Sequence features**
- Tick-level events with:
  - **Price**
  - **Size**
  - **Event type:** `add`, `cancel`, `trade`, `modify`
  - **Order rank** and other microstructural metadata

---

## 🎯 Task Definition

> **Predict the forward mid-price (10 seconds ahead)**  
> using past LOB and event sequence information.

---

## 🧩 Baseline Model — *Skew Signal*

As a benchmark, a **handcrafted skew signal** was designed:
- For each timestamp, compute **weighted mid-price**.
- Aggregate using **exponential time-decay weights**.
- Use this weighted mid as the predictor for forward mid-price.

**Performance:**

| Metric | Value |
|:--------|:-------:|
| Correlation (Final Alpha vs Target) | **0.07341** (≈7.3%) |

---

## 📚 Research References

This work is inspired by and extends upon these foundational papers:

1. [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books (Zhang et al., 2018)](https://arxiv.org/abs/1808.03668)
2. [Forecasting Stock Prices from LOB with Deep Learning (Tsantekidis et al., 2017)](https://arxiv.org/abs/1712.00975)
3. [Temporal Attention Augmented Bilinear Network (TABL) (Tran et al., 2021)](https://arxiv.org/abs/2102.08811)

---

## ⚙️ Model Experiments

### 1. **DeepLOB**

- Input: LOB data (bid/ask price and size)
- Architecture: CNN + Inception module + LSTM
- Experimented with:
  - Increasing convolutional filters and depth
  - Dropout regularization (sensitive hyperparameter)

**Results:**

| Metric | Value |
|:--------|:-------:|
| Validation Loss | 26.44906 |
| MAE | 3.10707 |
| Correlation | **0.09873** |
| R² | 0.00925 |

---

### 2. **Enhanced TABL (Temporal Attention Augmented Bilinear Network)**

- Extended TABL architecture with **multi-headed attention**.
- Fewer parameters than DeepLOB, yet improved correlation.
- Explored combinations of **LOB features** and **event sequence features**.

**Results:**

| Input Features | Val Loss | MAE | Corr. | R² |
|:----------------|:----------:|:-----:|:------:|:----:|
| Only OB Alpha features | 26.50047 | 3.11124 | 0.09641 | 0.00733 |
| Only Sequence Alpha features | 26.50071 | 3.11351 | 0.09656 | 0.00732 |
| Combined (trained separately, then merged) | 26.47405 | 3.10693 | 0.10995 | 0.00832 |
| **All features (single model)** | **26.40126** | **3.10690** | **0.10789** | **0.01104** |

---

## 🧠 Observations

- **DeepLOB** establishes a strong baseline using only LOB price/size data.  
- The **Enhanced TABL** with attention achieves better results with fewer parameters.  
- Including **event sequence data** improves generalization and correlation further.  
- **Best model performance:**  
  - **Correlation:** 0.108  
  - **R²:** 0.011  

---

## 📈 Summary of Results

| Model | Input Features | Correlation | R² |
|:--------|:----------------|:-------------:|:----:|
| Skew Signal (Baseline) | Weighted mid exponential | 0.073 | — |
| DeepLOB | LOB (price, size) | 0.099 | 0.009 |
| Enhanced TABL | LOB + Event Sequence | **0.108** | **0.011** |

---

## 🧮 Repository Structure
