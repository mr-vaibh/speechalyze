import os
import json
import nltk
import numpy as np
import sounddevice as sd
import librosa
import soundfile as sf
import opensmile
import parselmouth
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from nltk.tokenize import word_tokenize
from datetime import datetime

nltk.download("punkt")

# Constants
SAMPLE_RATE = 22050
DURATION = 60  # Recording duration in seconds
AUDIO_DIR = "audio"
WAV_DIR = os.path.join(AUDIO_DIR, "wav")

# Create directories if they don't exist
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

if not os.path.exists(WAV_DIR):
    os.makedirs(WAV_DIR)

# ---------------------- AUDIO PROCESSING ---------------------- #
def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    print("Recording complete.")
    return audio.flatten(), sample_rate

def load_audio(file_path, sample_rate=SAMPLE_RATE):
    if not os.path.exists(file_path):
        print("Error: File not found!")
        return None, None

    print("Loading MP3 file...")
    sound = AudioSegment.from_mp3(file_path)
    wav_path = os.path.join(WAV_DIR, os.path.basename(file_path.replace('.mp3', '.wav')))
    sound.export(wav_path, format="wav")
    audio, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    return audio[:sr * DURATION], sr

# ---------------------- FEATURE EXTRACTION ---------------------- #
def extract_jitter_shimmer(audio_path):
    sound = parselmouth.Sound(audio_path)
    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)")
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0.02, 0.0001, 1.3, 1.6)
    shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0.02, 0.0001, 1.3, 1.6, 1.3)
    return jitter, shimmer

def extract_jitter_shimmer_opensmile(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    features = smile.process_signal(y, sr)
    jitter = features["jitterLocal_sma3nz_amean"].values[0]  
    shimmer = features["shimmerLocaldB_sma3nz_amean"].values[0]
    return jitter, shimmer

def extract_wpm_and_pause_duration(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    sound = AudioSegment.from_wav(audio_path)
    nonsilent_ranges = detect_nonsilent(sound, min_silence_len=500, silence_thresh=-40)
    total_duration = len(sound) / 1000
    speech_duration = sum([(end - start) / 1000 for start, end in nonsilent_ranges])
    pause_duration = total_duration - speech_duration
    words_per_minute = (speech_duration / 0.3) / (total_duration / 60)
    return words_per_minute, pause_duration

# ---------------------- SPEECH ANALYSIS ---------------------- #
def transcribe_speech(audio, sr):
    model = Model(lang="en-us")
    recognizer = KaldiRecognizer(model, sr)
    audio_int16 = (audio * 32767).astype(np.int16).tobytes()
    recognizer.AcceptWaveform(audio_int16)
    result_json = recognizer.FinalResult()
    result = json.loads(result_json)
    transcribed_text = result.get("text", "")
    words_data = result.get("result", [])
    return transcribed_text, words_data

def analyze_text(text, duration=DURATION):
    words = word_tokenize(text)
    word_count = len(words)
    speech_rate = (word_count * 60) / duration
    word_freq = nltk.FreqDist(words)
    repeated_words = {word: count for word, count in word_freq.items() if count > 1}
    return speech_rate, repeated_words

# ---------------------- MORE FEATURES ---------------------- #
def extract_pitch_variability(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitch, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'))
    pitch_variability = np.std(pitch[voiced_flag]) if np.any(voiced_flag) else 0  # Standard deviation of pitch
    return pitch_variability

def extract_formants(audio_path):
    sound = parselmouth.Sound(audio_path)
    formants = parselmouth.praat.call(sound, "To Formant (burg)", 0.0, 50, 500, 0.01, 50)
    formant_values = [parselmouth.praat.call(formants, "Get value at time", 0.0, formant_index+1) for formant_index in range(3)]
    return formant_values  # Returns [F1, F2, F3]

def extract_syllable_duration(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    syllable_durations = librosa.frames_to_time(onset_frames, sr=sr)
    avg_syllable_duration = np.mean(np.diff(syllable_durations)) if len(syllable_durations) > 1 else 0
    return avg_syllable_duration

def extract_rhythm_and_tempo(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    inter_onset_intervals = np.diff(librosa.frames_to_time(onset_frames, sr=sr))
    mean_beat_interval = np.mean(inter_onset_intervals) if len(inter_onset_intervals) > 0 else 0
    return mean_beat_interval


# ---------------------- MAIN EXECUTION ---------------------- #
print("Choose input method:")
print("1 - Load an MP3 file")
print("2 - Record audio from the microphone (60 seconds)")
choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    file_path = input("Enter the MP3 file path: ").strip()
    audio, sr = load_audio(file_path)
    wav_path = os.path.join(WAV_DIR, os.path.basename(file_path.replace('.mp3', '.wav')))
    sf.write(wav_path, audio, sr)
elif choice == "2":
    audio, sr = record_audio(duration=DURATION)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"recorded_audio_{timestamp}.wav"
    wav_path = os.path.join(WAV_DIR, wav_filename)
    sf.write(wav_path, audio, sr)
else:
    print("Invalid choice. Exiting.")
    exit()

if audio is None:
    print("Failed to load or record audio. Exiting.")
    exit()

# Perform analysis
jitter, shimmer = extract_jitter_shimmer_opensmile(wav_path)
wpm, pause_duration = extract_wpm_and_pause_duration(wav_path)
transcribed_text, words_data = transcribe_speech(audio, sr)
speech_rate, repeated_words = analyze_text(transcribed_text)

# Additional analyses
pitch_variability = extract_pitch_variability(wav_path)
# formants = extract_formants(wav_path)
avg_syllable_duration = extract_syllable_duration(wav_path)
mean_beat_interval = extract_rhythm_and_tempo(wav_path)

# Display results
print(f"Jitter: {jitter:.4f}%")
print(f"Shimmer: {shimmer:.4f}%")
print(f"Words Per Minute: {wpm:.2f}")
print(f"Total Pause Duration: {pause_duration:.2f} sec")
print(f"Speech Rate: {speech_rate:.4f} words/min")
print(f"Repeated Words: {repeated_words}")
print(f"Transcribed Text: {transcribed_text}")

print(f"Pitch Variability (Jitter): {pitch_variability:.4f}")
# print(f"Formants (F1, F2, F3): {formants}")
print(f"Average Syllable Duration: {avg_syllable_duration:.4f} sec")
print(f"Mean Beat Interval (Rhythm/Tempo): {mean_beat_interval:.4f} sec")