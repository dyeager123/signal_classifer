import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib.mlab import specgram
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def generate_spectrogram_array(filename):
    sample_rate, samples = wavfile.read(filename)
    if samples.ndim > 1:
        samples = samples[:, 0]
    S, freqs, times = specgram(samples, NFFT=1024, Fs=sample_rate, noverlap=512)
    return S, freqs, times

def chop_spectrogram(S, times, segment_length_sec):
    total_duration_sec = times[-1]
    time_per_frame = times[1] - times[0]
    frames_per_segment = int(segment_length_sec / time_per_frame)
    num_segments = len(times) // frames_per_segment

    chopped_segments = []
    segment_start_times = []

    for i in range(num_segments):
        start_idx = i * frames_per_segment
        end_idx = start_idx + frames_per_segment
        segment = S[:, start_idx:end_idx]
        chopped_segments.append(segment)
        segment_start_times.append(times[start_idx])

    chopped_time = num_segments * segment_length_sec
    leftover = total_duration_sec - chopped_time
    print(f"Chopped into {num_segments} segments of {segment_length_sec} seconds.")
    if leftover > 0:
        print(f"⚠️ Discarded {leftover:.2f} seconds from the end.")

    return chopped_segments, segment_start_times

def flatten_segments(segments, to_db=True):
    flattened = []
    for segment in segments:
        if to_db:
            segment = 10 * np.log10(np.maximum(segment, 1e-10))
        flattened.append(segment.flatten())
    return np.array(flattened)

def apply_pca(data_matrix, n_components=None, use_variance_threshold=False, threshold=0.95):
    if use_variance_threshold:
        print(f"\nUsing PCA to retain {threshold*100:.1f}% of variance...")
        pca = PCA(n_components=threshold)
    else:
        print(f"\nApplying PCA to reduce to {n_components} dimensions...")
        pca = PCA(n_components=n_components)

    reduced = pca.fit_transform(data_matrix)
    explained = pca.explained_variance_ratio_

    print(f"✅ PCA complete. Total components used: {pca.n_components_}")
    print(f"Explained variance by component:")
    for i, var in enumerate(explained):
        print(f"  PC{i+1}: {var:.4f}")

    return reduced, pca

def apply_dbscan(data, eps=2.0, min_samples=2):
    print("\nRunning DBSCAN clustering...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)
    return labels

def print_clusters_and_anomalies(labels, start_times):
    from collections import defaultdict

    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[label].append(start_times[i])

    print("\n🧠 Clustering Summary:")
    for label, timestamps in cluster_map.items():
        if label == -1:
            print(f"\n🚨 Anomalous Segments (label = -1):")
        else:
            print(f"\nCluster {label}:")
        for t in timestamps:
            print(f"  → Segment starting at {t:.2f} sec")
        

# === Main logic ===
if __name__ == "__main__":
    FILENAME = "1_min_variable_sine.wav"
    SEGMENT_LENGTH_SEC = 5

    # PCA config
    USE_VARIANCE_THRESHOLD = False # True to use variance threshold, False for fixed components
    PCA_COMPONENTS = 3
    VARIANCE_THRESHOLD = 0.95

    # DBSCAN config
    DBSCAN_EPS = 2.0
    DBSCAN_MIN_SAMPLES = 2

    # Process pipeline
    S, freqs, times = generate_spectrogram_array(FILENAME)
    segments, start_times = chop_spectrogram(S, times, SEGMENT_LENGTH_SEC)
    data_matrix = flatten_segments(segments)

    print(f"\nOriginal data matrix shape: {data_matrix.shape}")
    reduced_matrix, pca_model = apply_pca(
        data_matrix,
        n_components=PCA_COMPONENTS,
        use_variance_threshold=USE_VARIANCE_THRESHOLD,
        threshold=VARIANCE_THRESHOLD
    )

    print(f"\nReduced matrix shape: {reduced_matrix.shape}")
    labels = apply_dbscan(reduced_matrix, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    print_clusters_and_anomalies(labels, start_times)
