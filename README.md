# Speech Defect Detection System

## Overview

This project aims to detect speech defects in individuals based on various acoustic and speech features. The system records or processes audio files, extracts relevant features, and uses machine learning to classify the severity of speech defects. It can handle both recorded speech and MP3 files as input and provides real-time predictions regarding the severity of speech defects.

### Key Features:
- **Speech Analysis**: Includes extraction of jitter, shimmer, words per minute (WPM), pause duration, pitch variability, formants, syllable duration, and rhythm/tempo.
- **Speech Defect Classification**: Uses a machine learning model trained on synthetic data to classify speech defects into various categories: No Defect, Low Defect, Mild Defect, Medium Defect, High Defect, and Severe Defect.
- **Model Training and Evaluation**: Implements multiple machine learning models (Logistic Regression, SVM, Decision Tree, Random Forest) and evaluates them using GridSearchCV to select the best model.
- **Real-Time Prediction**: Allows the user to input features of their speech for real-time defect classification.

## Structure

The project is divided into three main Python scripts:

1. **`speech-features.py`**:
   - Responsible for recording and processing audio.
   - Extracts acoustic features like jitter, shimmer, pitch, formants, syllable duration, etc.
   - Uses libraries like `librosa`, `opensmile`, `parselmouth`, and `vosk` for feature extraction and speech-to-text.

2. **`generate_speech_defect_data.py`**:
   - Generates synthetic speech defect data with balanced categories.
   - Saves the data to a CSV file (`synthetic_speech_defect_data.csv`) which is used for training the machine learning model.

3. **`speech-ml.py`**:
   - Handles the machine learning pipeline.
   - Trains multiple classifiers on the generated data and evaluates them.
   - Allows the user to make predictions on speech features and classify the severity of defects.

## Setup and Installation

### Prerequisites

You need Python 3.7 or higher to run this project. Additionally, the following libraries must be installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `librosa`
- `nltk`
- `sounddevice`
- `soundfile`
- `opensmile`
- `parselmouth`
- `pydub`
- `vosk`

You can install the required dependencies by running:

```bash
pip install numpy pandas scikit-learn librosa nltk sounddevice soundfile opensmile parselmouth pydub vosk
```

### Running the Project

1. **Generate the Synthetic Speech Data**:
   First, run the `generate_speech_defect_data.py` script to generate the synthetic dataset used for training.

   ```bash
   python generate_speech_defect_data.py
   ```

   This will generate a file called `synthetic_speech_defect_data.csv` in the `data` directory.

2. **Train the Machine Learning Model**:
   After generating the data, run the `speech-ml.py` script to train the machine learning model on the generated dataset.

   ```bash
   python speech-ml.py
   ```

   This will train several machine learning models, evaluate their accuracy, and display the results. It will also allow you to interact with the model and make predictions.

3. **Audio Recording and Analysis**:
   If you want to analyze your own speech, you can use the `speech-features.py` script. This script allows you to either load an MP3 file or record audio using your microphone. It will then extract the relevant features from the speech and display the results.

   ```bash
   python speech-features.py
   ```

   - **Option 1**: Load an MP3 file and analyze it.
   - **Option 2**: Record audio from your microphone (default duration of 60 seconds).

## Input and Output

### Input:
- **Audio**: You can either record audio directly from the microphone or provide an MP3 file for analysis.
- **Speech Features**: The system extracts various speech features including jitter, shimmer, WPM, pause duration, pitch variability, formants, syllable duration, and rhythm/tempo.

### Output:
- **Feature Analysis**: The script provides detailed results about the audio file or recording:
  - Jitter and shimmer values
  - Words per minute (WPM)
  - Total pause duration
  - Transcribed text
  - Repeated words
  - Pitch variability
  - Formant frequencies
  - Average syllable duration
  - Mean beat interval (Rhythm/Tempo)

- **Speech Defect Classification**: Based on the extracted features, the system classifies the severity of the speech defect into one of the following categories:
  - No Defect
  - Low Defect
  - Mild Defect
  - Medium Defect
  - High Defect
  - Severe Defect

### Example of output:
```
Jitter: 0.2032%
Shimmer: 0.5432%
Words Per Minute: 128.57
Total Pause Duration: 2.30 sec
Speech Rate: 160.28 words/min
Repeated Words: {'hello': 3, 'how': 2}
Transcribed Text: "Hello, how are you today?"
Pitch Variability (Jitter): 0.1623
Formants (F1, F2, F3): [560.0, 1500.0, 2500.0]
Average Syllable Duration: 0.1102 sec
Mean Beat Interval (Rhythm/Tempo): 0.2400 sec
```

## Machine Learning Model

The system uses a classification approach to detect speech defects. It is trained using synthetic data that simulates various levels of defects based on features like jitter, shimmer, WPM, and pause duration. The following models are evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier

The model that performs best in terms of accuracy is used for making real-time predictions on user-provided data.

## How to Use

1. Run the training script (`speech-ml.py`) to train the model.
2. Once the model is trained, you can use the same script to make predictions based on user input:
   - Enter features like jitter, shimmer, WPM, and pause duration when prompted.
   - The system will predict the severity of the speech defect.

### Example:

```
Enter Jitter (%): 1.2
Enter Shimmer (%): 1.5
Enter Words per Minute: 110
Enter Pauses Duration (sec): 3.5

Predicted Severity of Defect: Mild Defect
```

## Future Enhancements
- Integrate more advanced speech defect detection features such as voice quality analysis.
- Improve the classification model with more real-world data.
- Implement a web interface for real-time interaction and analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thank you for using this Speech Defect Detection System!