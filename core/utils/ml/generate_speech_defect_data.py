import os
import numpy as np
import pandas as pd

# Constants
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def classify_severity(jitter, shimmer, wpm, pauses):
    """Classify the severity of defect based on jitter, shimmer, WPM, and pauses."""
    if jitter <= 0.5 and shimmer <= 1 and wpm >= 120 and pauses < 1:
        return "No Defect"
    elif 0.5 < jitter <= 1 and 1 < shimmer <= 2 and 100 <= wpm < 120 and 1 <= pauses < 2:
        return "Low Defect"
    elif 1 < jitter <= 2 and 2 < shimmer <= 3 and 85 <= wpm < 100 and 2 <= pauses < 4:
        return "Mild Defect"
    elif 2 < jitter <= 3 and 3 < shimmer <= 4 and 60 <= wpm < 85 and 3 <= pauses < 5:
        return "Medium Defect"
    elif 3 < jitter <= 4 and 4 < shimmer <= 6 and 40 <= wpm < 60 and 5 <= pauses < 10:
        return "High Defect"
    else:
        return "Severe Defect"

# Generate random data with balanced distribution
np.random.seed(42)  # For reproducibility
num_samples = 10000

def generate_balanced_data(num_samples):
    categories = {
        "No Defect": (0, 0.5, 0, 1, 120, 150, 0, 1),
        "Low Defect": (0.5, 1, 1, 2, 100, 120, 1, 2),
        "Mild Defect": (1, 2, 2, 3, 85, 100, 2, 4),
        "Medium Defect": (2, 3, 3, 4, 60, 85, 3, 5),
        "High Defect": (3, 4, 4, 6, 40, 60, 5, 10),
        "Severe Defect": (4, 5.5, 6, 7.5, 20, 40, 10, 15)
    }
    
    samples_per_category = num_samples // len(categories)
    data = []
    
    for severity, (j_min, j_max, s_min, s_max, w_min, w_max, p_min, p_max) in categories.items():
        for _ in range(samples_per_category):
            jitter = np.random.uniform(j_min, j_max)
            shimmer = np.random.uniform(s_min, s_max)
            wpm = np.random.uniform(w_min, w_max)
            pauses = np.random.uniform(p_min, p_max)
            data.append([jitter, shimmer, wpm, pauses, severity])
    
    return pd.DataFrame(data, columns=["Jitter (%)", "Shimmer (%)", "WPM (Words per Minute)", "Pauses (Duration in sec)", "Severity of Defect"])

# Generate balanced dataset
df = generate_balanced_data(num_samples)

# Save to CSV
df.to_csv("data/synthetic_speech_defect_data.csv", index=False)

print("Balanced dataset generated and saved as 'speech_defect_data.csv'")
