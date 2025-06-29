import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

class SpectrogramGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Anomaly Detection")

        self.filename = None
        self.audio_data = None
        self.sample_rate = None
        self.S = None
        self.freqs = None
        self.times = None

        # === GUI Layout ===
        self.load_button = tk.Button(master, text="Load WAV File", command=self.load_file)
        self.load_button.pack()

        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack()

        self.segment_label = tk.Label(master, text="Segment Length (seconds):")
        self.segment_label.pack()
        self.segment_slider = tk.Scale(master, from_=1, to=20, orient=tk.HORIZONTAL)
        self.segment_slider.set(5)
        self.segment_slider.pack()

        self.pca_option = tk.StringVar(value="pca_3")
        self.pca_3_radio = tk.Radiobutton(master, text="PCA with 3 Components", variable=self.pca_option, value="pca_3")
        self.pca_95_radio = tk.Radiobutton(master, text="PCA with 95% Variance", variable=self.pca_option, value="pca_95")
        self.pca_3_radio.pack()
        self.pca_95_radio.pack()

        self.run_button = tk.Button(master, text="Run Analysis", command=self.run_analysis)
        self.run_button.pack()

    def load_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if not self.filename:
            return

        self.sample_rate, samples = wavfile.read(self.filename)
        if samples.ndim > 1:
            samples = samples[:, 0]
        self.audio_data = samples

        self.S, self.freqs, self.times = specgram(
            self.audio_data, NFFT=1024, Fs=self.sample_rate, noverlap=512
        )

        self.show_spectrogram()

    def show_spectrogram(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(10 * np.log10(self.S), aspect="auto", origin="lower", cmap="viridis",
                  extent=[self.times[0], self.times[-1], self.freqs[0], self.freqs[-1]])
        ax.set_title("Original Spectrogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()

    def chop_spectrogram(self, segment_length_sec):
        time_per_frame = self.times[1] - self.times[0]
        frames_per_segment = int(segment_length_sec / time_per_frame)
        num_segments = len(self.times) // frames_per_segment

        chopped_segments = []
        segment_start_times = []

        for i in range(num_segments):
            start = i * frames_per_segment
            end = start + frames_per_segment
            chopped_segments.append(self.S[:, start:end])
            segment_start_times.append(self.times[start])

        return chopped_segments, segment_start_times

    def flatten_segments(self, segments):
        return np.array([10 * np.log10(np.maximum(seg, 1e-10)).flatten() for seg in segments])

    def run_analysis(self):
        segment_length = self.segment_slider.get()
        pca_mode = self.pca_option.get()

        segments, start_times = self.chop_spectrogram(segment_length)
        data_matrix = self.flatten_segments(segments)

        if pca_mode == "pca_3":
            pca = PCA(n_components=3)
        else:
            pca = PCA(n_components=0.95)

        reduced = pca.fit_transform(data_matrix)

        # DBSCAN
        db = DBSCAN(eps=2.0, min_samples=2)
        labels = db.fit_predict(reduced)

        # Report anomalies
        print("\n🚨 Anomalous Segments (Label = -1):")
        for i, label in enumerate(labels):
            if label == -1:
                print(f"  → Segment starting at {start_times[i]:.2f} sec")

        if pca_mode == "pca_3":
            self.plot_3d_clusters(reduced, labels, start_times)

    def plot_3d_clusters(self, data, labels, start_times):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = set(labels)
        colors = plt.cm.get_cmap("tab10", len(unique_labels))

        for label in unique_labels:
            idxs = np.where(labels == label)[0]
            points = data[idxs]
            label_text = "Anomaly" if label == -1 else f"Cluster {label}"
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=label_text)

        ax.set_title("DBSCAN Clustering (PCA-3D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.tight_layout()
        plt.show()

# === Run the App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrogramGUI(root)
    root.mainloop()
