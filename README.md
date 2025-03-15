# Speechalyze - Speech Defect Detection System

## Overview

This project aims to detect speech defects in individuals based on various acoustic and speech features. The system records or processes audio files, extracts relevant features, and uses machine learning to classify the severity of speech defects. It can handle both recorded speech and MP3 files as input and provides real-time predictions regarding the severity of speech defects.

### Key Features:
- **Speech Analysis**: Includes extraction of jitter, shimmer, words per minute (WPM), pause duration, pitch variability, formants, syllable duration, and rhythm/tempo.
- **Speech Defect Classification**: Uses a machine learning model trained on synthetic data to classify speech defects into various categories: No Defect, Low Defect, Mild Defect, Medium Defect, High Defect, and Severe Defect.
- **Model Training and Evaluation**: Implements multiple machine learning models (Logistic Regression, SVM, Decision Tree, Random Forest) and evaluates them using GridSearchCV to select the best model.
- **Real-Time Prediction**: Allows the user to input features of their speech for real-time defect classification.

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