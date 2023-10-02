import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your dataset (assuming it's in a CSV file)
data = pd.read_csv('C:\\Users\\Shlok\\Downloads\\Student_mental_normalized2.csv')

# Data preprocessing
# Assuming 'diagnosis' is the target variable
columns_to_exclude = ['id', 'Diagnosis_text', 'Diagnosis']

# Exclude highly correlated columns as needed
X = data.drop(columns=columns_to_exclude, axis=1)
y = data['Diagnosis']

# Perform k-fold cross-validation for different values of n_estimators
n_estimators_values = [10]  # Specify a range of values
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for n_estimators in range(10,14):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    scores = cross_val_score(model, X, y, cv=k_fold, scoring='accuracy')
    cv_scores.append(scores.mean())

# Choose the n_estimators value with the highest cross-validated accuracy
best_n_estimators = n_estimators_values[np.argmax(cv_scores)]

# Train the final model with the best n_estimators value on the entire dataset
final_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
final_model.fit(X, y)

# Evaluate the final model using k-fold cross-validation
accuracy = np.mean(cross_val_score(final_model, X, y, cv=k_fold, scoring='accuracy'))
print(f"Best n_estimators: {best_n_estimators}")
print(f"Mean accuracy using k-fold cross-validation: {accuracy:.2f}")
