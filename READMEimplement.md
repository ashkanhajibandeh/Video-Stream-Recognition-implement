# Video-Stream Recognition (PyTorch demo)
This repository contains a full end-to-end PyTorch demo that implements the method from "Video Stream Recognition Using Bitstream Shape for Mobile Network QoE" (Chmieliauskas & Paulikas, Sensors 2023). The demo uses a synthetic dataset that mimics the paper's characteristics (HAS periodic chunks, other apps bursts, background).

## Files produced
- results/sample_bitstreams.png
- results/kde_stats.png
- results/fundamental_cdf.png
- results/accuracy_curve.png
- results/confusion_matrix.png
- results/classification_report.txt
- results/best_model.pth
- results/X300.npy, results/Y300.npy

## How to run
1. Create a Python environment with: Python 3.8+, torch, numpy, matplotlib, scipy, scikit-learn, seaborn, tqdm
2. Run:
   python video_recognition.py --outdir results --epochs 12 --batch_size 64

## What the script does
1. Generates synthetic bitstreams for three classes: video (HAS-like), other apps, and start-screen/background.
2. Builds windows of 300s (also generates 60s and 600s sets for analysis).
3. Trains a Conv1D network (downlink+uplink channels).
4. Saves model, metrics and plots for presentation.

## Files from the paper to include in presentation (cut from the PDF)
Use the following figures/tables from the article for direct comparison:
- Figure 2 (sample bitstreams). :contentReference[oaicite:9]{index=9}
- Figure 3 (KDE mean/std). :contentReference[oaicite:10]{index=10}
- Figure 4 (Fundamental frequency CDF). :contentReference[oaicite:11]{index=11}
- Figure 5 (CNN architecture diagram). :contentReference[oaicite:12]{index=12}
- Table 2 (Test Accuracy). :contentReference[oaicite:13]{index=13}
- Table 3 (Precision and Recall). :contentReference[oaicite:14]{index=14}
- Figure 6 (Misclassification by app). :contentReference[oaicite:15]{index=15}

## Notes and next steps
- The script uses synthetic data. To reproduce paper-level accuracy, collect a real dataset similar to the authors (Android app logging downlink/uplink and active app) and replace dataset generation with loading that data.
- For final experiments use GPU and increase epochs and dataset size.
