import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Read the data from the CSV file
df = pd.read_csv('data/synthetic_speech_defect_data.csv')

# Strip any leading/trailing spaces in the column names (if any)
df.columns = df.columns.str.strip()

# Split the data into features (X) and target (y)
X = df[['Jitter (%)', 'Shimmer (%)', 'WPM (Words per Minute)', 'Pauses (Duration in sec)']]
y = df['Severity of Defect']

# Encode the target variable (Severity of Defect)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert to numeric labels

# Preprocessing: No need for one-hot encoding on features (all are numeric)
# So we will just scale them using a StandardScaler
from sklearn.preprocessing import StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Jitter (%)', 'Shimmer (%)', 'WPM (Words per Minute)', 'Pauses (Duration in sec)'])
    ]
)

# Function to create a pipeline and evaluate models
def create_model_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# Models to compare (classification models)
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
}

# Parameter grid for GridSearchCV (for models that benefit from it)
param_grids = {
    'Logistic Regression': {'classifier__C': [0.1, 1, 10]},
    'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']},
    'Decision Tree': {'classifier__max_depth': [5, 10, 15, 20], 'classifier__min_samples_split': [2, 5, 10]},
    'Random Forest': {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [5, 10, 20]},
}

# Initialize result list
results = []

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # Create pipeline for the current model
    pipeline = create_model_pipeline(model)

    # If using GridSearchCV, apply it to find the best parameters
    grid_search = GridSearchCV(pipeline, param_grids.get(model_name, {}), cv=5, scoring='accuracy', n_jobs=-1)
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model to the training data using GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Get the best estimator and its performance on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the results for this model
    results.append({
        'Model': model_name,
        'Best Parameters': grid_search.best_params_,
        'Accuracy': accuracy
    })

# Create a DataFrame to display results
results_df = pd.DataFrame(results)

# Display the results in a neat DataFrame
print("\nModel Comparison Results:")
print(results_df)

# Select the best model (you can choose based on accuracy, for example)
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
print(f"\nBest Model: {best_model_name}")

# Create and fit the best model pipeline
best_model_pipeline = create_model_pipeline(models[best_model_name])
best_model_pipeline.fit(X_train, y_train)

# Now, run a while loop to get user input and make predictions
while True:
    # Get user input
    jitter = float(input("\nEnter Jitter (%): "))
    shimmer = float(input("Enter Shimmer (%): "))
    wpm = float(input("Enter Words per Minute: "))
    pauses = float(input("Enter Pauses Duration (sec): "))

    # Create a DataFrame for input
    user_input = pd.DataFrame([[jitter, shimmer, wpm, pauses]], columns=['Jitter (%)', 'Shimmer (%)', 'WPM (Words per Minute)', 'Pauses (Duration in sec)'])

    # Make a prediction
    predicted_class = best_model_pipeline.predict(user_input)[0]
    
    # Decode the prediction back to the original label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    # Display the result
    print(f"Predicted Severity of Defect: {predicted_label}")

    # Ask if the user wants to make another prediction
    continue_input = input("\nDo you want to make another prediction? (y/n): ").lower()
    if continue_input != 'y':
        break

print("Goodbye!")
