import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Use RandomForestClassifier for classification tasks
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the lung cancer dataset
data = pd.read_csv('survey_lung_cancer.csv')  # Use your actual file name

# Data preprocessing (encoding categorical variables)
data['GENDER'] = data['GENDER'].astype('category').cat.codes  # Encode 'GENDER' as 0 (F) and 1 (M)
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})  # Encode target variable

# Prepare feature and target variables
X = data[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE',
          'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
          'SWALLOWING DIFFICULTY', 'CHEST PAIN']]  # Features
y = data['LUNG_CANCER']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Use RandomForestClassifier
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save the model
joblib.dump(model, 'lung_cancer_model.pkl')
