import os
import json
import random
import string
import nltk
import numpy as np
import librosa
import opensmile
import parselmouth
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")

# Constants
SAMPLE_RATE = 22050
DURATION = 60  # Recording duration in seconds
AUDIO_DIR = "core/utils/speech/audio/"
WAV_DIR = os.path.join(AUDIO_DIR, "wav")

# Create directories if they don't exist
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

if not os.path.exists(WAV_DIR):
    os.makedirs(WAV_DIR)

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

def extract_features(audio_file, input="file"):
    # Ensure the WAV directory exists
    if not os.path.exists(WAV_DIR):
        os.makedirs(WAV_DIR)

    # Generate a random name for the WAV file
    random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + ".wav"
    wav_path = os.path.join(WAV_DIR, random_filename)

    # Check if the audio file is already in .wav format
    if audio_file.name.endswith('.wav'):
        # If it's already a .wav file, directly save it to the WAV directory
        with open(wav_path, "wb") as f:
            f.write(audio_file.read())
    else:
        # Read the content of the uploaded audio file (InMemoryUploadedFile)
        audio_data = audio_file.read()  # This gets the byte content of the uploaded file
        
        # Save the byte data to a temporary MP3 file
        temp_mp3_path = os.path.join(AUDIO_DIR, f"{random_filename}.mp3")
        with open(temp_mp3_path, "wb") as temp_file:
            temp_file.write(audio_data)

        # Convert MP3 to WAV using pydub
        sound = AudioSegment.from_mp3(temp_mp3_path)
        sound.export(wav_path, format="wav")

        # Delete the temporary MP3 file
        os.remove(temp_mp3_path)
        
    # Load the audio and proceed with feature extraction
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    audio = audio[:sr * DURATION]

    if audio is None:
        print("Failed to load or record audio. Exiting.")
        exit()

    jitter, shimmer = extract_jitter_shimmer_opensmile(wav_path)
    wpm, pause_duration = extract_wpm_and_pause_duration(wav_path)
    transcribed_text, words_data = transcribe_speech(audio, sr)
    speech_rate, repeated_words = analyze_text(transcribed_text)
    pitch_variability = extract_pitch_variability(wav_path)
    avg_syllable_duration = extract_syllable_duration(wav_path)
    mean_beat_interval = extract_rhythm_and_tempo(wav_path)

    # Delete the temporary WAV file
    os.remove(wav_path)

    # Return all the extracted features
    return {
        "jitter": jitter,
        "shimmer": shimmer,
        "words_per_minute": wpm,
        "pause_duration": pause_duration,
        "speech_rate": speech_rate,
        "repeated_words": repeated_words,
        "pitch_variability": pitch_variability,
        "avg_syllable_duration": avg_syllable_duration,
        "mean_beat_interval": mean_beat_interval,
        "transcribed_text": transcribed_text,
    }