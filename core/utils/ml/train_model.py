import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Base directory for the ML module
BASE_ML_DIR = 'core/utils/ml'

# Read the data from the CSV file
df = pd.read_csv(f'{BASE_ML_DIR}/data/synthetic_speech_defect_data.csv')

# Strip any leading/trailing spaces in the column names (if any)
df.columns = df.columns.str.strip()

# Split the data into features (X) and target (y)
X = df[['Jitter (%)', 'Shimmer (%)', 'WPM (Words per Minute)', 'Pauses (Duration in sec)']]
y = df['Severity of Defect']

# Encode the target variable (Severity of Defect)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert to numeric labels

# Preprocessing: Scaling features using StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Jitter (%)', 'Shimmer (%)', 'WPM (Words per Minute)', 'Pauses (Duration in sec)'])
    ]
)

# Create a pipeline for Logistic Regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Parameter grid for Logistic Regression
param_grid = {
    'classifier__C': [0.1, 1, 10]
}

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply GridSearchCV to tune the hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy and best parameters
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy}")

# Save the model using joblib
joblib.dump(best_model, f'{BASE_ML_DIR}/model/model_speech_defect_logistic_regression.pkl')
joblib.dump(label_encoder, f'{BASE_ML_DIR}/model/label_speech_defect_logistic_regression.pkl')

print("\nModel saved successfully in 'core/utils/ml/model' directory.")
print("Label encoder saved successfully in 'core/utils/ml/model' directory.")