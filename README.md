# 🌀 Time-Series Audio Clustering with DBSCAN (v0.1.0)

**Version 0.1.0** – Published as a **demo and early prototype**. This project explores dimensionality reduction and non-parametric clustering applied to time series data, with an emphasis on audio frequency signals.

---

## 🔍 Overview

This tool processes `.wav` files by:
1. Translating audio into a **spectrogram**.
2. Splitting the spectrogram into **equally spaced time segments**.
3. Performing **PCA** for dimensionality reduction.
4. Applying **DBSCAN** to detect clusters and anomalies in the signal.

The approach is generalizable to any time series where **frequency is a recurring, meaningful feature**.

---

## ▶️ How to Use

### 📄 `DBSCAN_outline.py`

Run this script with any `.wav` file to:
- Convert it into a spectrogram
- Reduce dimensionality using PCA (either 95% threshold or fixed component count)
- Cluster the segments using DBSCAN

**Note:** You must manually edit the script to:
- Set the path to your `.wav` file
- Adjust PCA or DBSCAN parameters

The script will then visualize clustered vs. anomalous segments.

---

## 🖼️ GUI Preview – `Gui_DBSCAN_sketch.py`

This is a **prototype GUI** representing the future direction of the project.

Features:
- Upload `.wav` files through a file explorer
- Display spectrograms interactively
- Select segment duration
- Choose PCA configuration (3 components or 95% threshold)
- Visualize results in 3D if using 3 PCA components
- Modify DBSCAN hyperparameters directly in the interface

A simple `.wav` file generator is also included for demo purposes.

---

## 🚀 Roadmap – Version 1.0 Goals

- ✅ Visually enhanced GUI using a modern GUI framework
- ✅ Support for additional data formats: `.csv`, `.xlsx`, etc.
- ✅ Options for advanced dimensionality reduction: `t-SNE`, `UMAP`, `ICA`, etc.
- ✅ Support for additional clustering methods: `HDBSCAN`, `Mean Shift`, `Spectral Clustering`
- ✅ In-GUI spectrogram previews per cluster with optional **average cluster spectrogram** computation

---

## 📜 License

This project is **open source** and free to use for research, experimentation, and development.

---

## 🙏 Thanks

Thanks for exploring this project. It’s part of my journey into GUI development and advanced, non-parametric data analysis techniques. Feedback, suggestions, or collaborations are welcome!

