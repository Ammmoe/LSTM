# Trajectory Prediction Training (`train.py`)

This repository contains a **sequence-to-coordinate trajectory prediction training script** written in PyTorch. It predicts **3D trajectories of a single drone (or agent)** by taking a sequence of past coordinates and predicting future positions.

---

## 🔁 Workflow (what `train.py` does)

1. Load or generate 3D trajectory data
2. Convert trajectories into sequences of past frames (`LOOK_BACK`) and future frames (`FORWARD_LEN`)
3. Split into training and testing sets
4. Normalize features and convert to PyTorch tensors
5. Train a sequence-to-sequence model (MSE loss, optional early stopping)
6. Evaluate performance (MSE, RMSE, MAE, EDE)
7. Visualize predictions vs ground truth in 3D plots
8. Save model, config, and plots

---

## 📦 Requirements (install first)

```bash
pip install -r requirements.txt
```

Minimum recommended environment:

* Python 3.8+
* PyTorch
* NumPy
* scikit-learn
* matplotlib

---

## ⚙️ Quick start — choose a model

Change the model import at the top of `train.py` to select the architecture you want to test:

```python
# Example: attention bi-directional GRU with attention
from models.attention_bi_gru_predictor import TrajPredictor
```

Available model modules (choose one and update the import):

1. `attention_bi_gru_predictor` — Bidirectional GRU with attention
2. `attention_bi_lstm_predictor` — Bidirectional LSTM with attention
3. `attention_gru_predictor` — Uni-directional GRU with attention
4. `attention_lstm_predictor` — Uni-directional LSTM with attention
5. `gru_predictor` — Plain GRU
6. `lstm_predictor` — Plain LSTM
7. `rnn_predictor` — Plain RNN

> ⚠️ **Important:** Bidirectional architectures require separate encoder/decoder hidden sizes (`enc_hidden_size`, `dec_hidden_size`). Uni-directional architectures typically use a single `hidden_size`.

---

## 🛠 Model parameters (set after choosing model)

**For bidirectional models:**

```python
model_params = {
    "input_size": 3,       # 3 features: x, y, z (or 4 if including time)
    "enc_hidden_size": 64, # encoder hidden size
    "dec_hidden_size": 64, # decoder hidden size
    "output_size": 3,      # same as input_size normally
    "num_layers": 1,
}
```

**For uni-directional models:**

```python
model_params = {
    "input_size": 3,   # x, y, z (or 4 if including time)
    "hidden_size": 64, # hidden size for encoder/decoder
    "output_size": 3,
    "num_layers": 1,
}
```

**Using time as an input feature:**

If you want to include timestep (t) as a feature, set:

```python
USE_TIME_FEATURE = True
```

Then update input/output sizes to `4`:

```python
model_params = {
    "input_size": 4,   # x, y, z, t
    "hidden_size": 64,
    "output_size": 4,
    "num_layers": 1,
}
```

---

## 📁 Dataset configuration

Set the dataset type in `train.py`:

```python
DATA_TYPE = "zurich"   # Options: "artificial", "quadcopter", "zurich"
```

* `zurich` — dataset cleaned to 10 Hz (0.1 s per step). `FORWARD_LEN=10` → 1 second ahead.
* `quadcopter` — real quadcopter trajectories.
* `artificial` — generated sine/cosine 3D trajectories for quick testing.

---

## ⏱️ Data & training parameters

```python
# Data parameters
LOOK_BACK  = 50  # number of past frames used as input
FORWARD_LEN = 5  # number of future frames to predict

# Training parameters
EPOCHS = 500
BATCH_SIZE = 70
LEARNING_RATE = 1e-3
```

* For Zurich (10 Hz): `FORWARD_LEN = 10` equals **1 second** into the future.
* Tweak `LOOK_BACK` and `FORWARD_LEN` to experiment with short-term vs long-term forecasting.

---

## 📊 Plotting & output

```python
NUM_PLOTS = 6  # number of example trajectories to visualize after training
```

After training the script will:

* Save the trained model file
* Save training configuration (JSON)
* Save training plots (3D predicted vs ground truth)
* Print evaluation metrics: **MSE, RMSE, MAE, EDE (Euclidean Distance Error)**

---

## ✅ Quick usage example

Change dataset, model, and include time feature in one place inside `train.py`:

```python
# dataset
DATA_TYPE = "zurich"

# model import (choose one)
from models.attention_bi_gru_predictor import TrajPredictor

# use timestep feature
USE_TIME_FEATURE = False

# model params for bidirectional + time
model_params = {
    "input_size": 3,
    "enc_hidden_size": 128,
    "dec_hidden_size": 128,
    "output_size": 3,
    "num_layers": 2,
}

# training and data
LOOK_BACK  = 50
FORWARD_LEN = 10  # 10 steps @10Hz => 1 second
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_PLOTS = 4
```

---

## ▶️ Run the training script

```bash
python train.py
# or
python3 train.py
```

---

## 📚 Dataset Credits

This project makes use of the following publicly available datasets:

* **Zurich MAV Dataset** – [https://rpg.ifi.uzh.ch/zurichmavdataset.html](https://rpg.ifi.uzh.ch/zurichmavdataset.html)  
* **Quadcopter Delivery Dataset (CMU)** – [https://kilthub.cmu.edu/articles/dataset/Data_Collected_with_Package_Delivery_Quadcopter_Drone/12683453](https://kilthub.cmu.edu/articles/dataset/Data_Collected_with_Package_Delivery_Quadcopter_Drone/12683453)

---

Happy experimenting — tweak parameters, try different architectures, and compare metrics/plots to find what works best.
