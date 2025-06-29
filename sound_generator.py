from pydub import AudioSegment
import numpy as np
import wave
import struct
import random

def generate_sine_wave_segments(duration_sec=60, sample_rate=44100, amplitude=0.5):
    total_samples = duration_sec * sample_rate
    samples = []

    time_elapsed = 0
    while time_elapsed < duration_sec:
        # Random segment length between 0.5 and 5 seconds
        segment_duration = random.uniform(0.5, 5.0)
        segment_duration = min(segment_duration, duration_sec - time_elapsed)  # Don't go past total

        frequency = random.choice([220, 330, 440, 550, 660, 880, 990])  # Random musical tone
        print(f"Adding {segment_duration:.2f}s of {frequency}Hz")

        num_segment_samples = int(segment_duration * sample_rate)
        t = np.arange(num_segment_samples)
        segment = amplitude * np.sin(2 * np.pi * frequency * t / sample_rate)
        samples.extend(segment)

        time_elapsed += segment_duration

    # Convert to 16-bit PCM
    samples = np.array(samples)
    samples = np.int16(samples * 32767)

    # Save as WAV file
    with wave.open("1_min_variable_sine.wav", "w") as wav_file:
        wav_file.setnchannels(1)          # mono
        wav_file.setsampwidth(2)          # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        for s in samples:
            wav_file.writeframes(struct.pack('<h', s))

    print("Saved as '1_min_variable_sine.wav'.")

generate_sine_wave_segments()