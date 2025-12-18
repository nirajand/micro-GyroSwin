# 🚀 Neural-Gyrokinetics-5D

> **Physics-Informed 5D Plasma Turbulence Prediction System**

This repository implements a scalable, production-ready machine learning pipeline for predicting heat flux in fusion plasmas. It utilizes a **Micro-GyroSwin Transformer** architecture optimized for hardware ranging from a **6GB RTX 2060** to multi-node HPC clusters.

---

## 🏗 Project Architecture

The system is divided into modular components to ensure memory isolation and scalability:

* **`utils.py`**: The Physics Engine. Contains the Maxwellian distribution laws and heat flux moment calculations.
* **`data_engine.py`**: Generates physics-consistent 5D synthetic datasets using Zarr for disk-streaming.
* **`model_factory.py`**: The "Blueprint." Uses PyTorch Lightning + Xformers for memory-efficient 5D attention.
* **`main.py`**: The Orchestrator. Handles training, validation, and hardware detection.

---

## ⚡ Quick Start (Hit and Run)

To execute the entire pipeline (Data Gen → Training → Inference) on your current hardware, simply run:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh

```

### Manual Usage

If you want to run specific stages:

1. **Generate Physics Data:**
```bash
python data_engine.py

```


2. **Train Model (Auto-detects GPU/CPU):**
```bash
python main.py --accel gpu

```



---

## 🧬 Laws of Physics Implemented

Unlike standard ML models, this pipeline is constrained by:

1. **Maxwell-Boltzmann Velocity Distribution**: Initial states are generated following f_0 \propto \exp(-\frac{E}{T}).
2. **Heat Flux Conservation**: Labels are derived from the 3rd velocity moment of the distribution function.
3. **5D Geometric Consistency**: The model preserves the spatial-velocity relationship (x, y, z, v_{\parallel}, \mu).

---

## 🛠 Scalability & Hardware Support

| Hardware | Optimization Strategy | Status |
| --- | --- | --- |
| **RTX 2060 (6GB)** | 16-bit Precision + Gradient Checkpointing + Fused Attention | ✅ Supported |
| **Multi-GPU (A100/H100)** | Distributed Data Parallel (DDP) | ✅ Supported |
| **TPU (v2/v3)** | PyTorch XLA Integration | ✅ Supported |
| **CPU Only** | 32-bit Standard Precision | ✅ Supported |

---

## 📊 Directory Structure

```text
.
├── data/               # Persistent Zarr datasets
├── models/             # Trained checkpoint files (.ckpt)
├── src/                # Core logic
│   ├── utils.py        # Physics definitions
│   ├── model_factory.py# Architecture
│   └── data_engine.py  # Data generation
├── main.py             # Entry point
└── run_pipeline.sh     # Automation script

```

---

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Would you like me to help you initialize this repository on your local machine using Git commands?**
