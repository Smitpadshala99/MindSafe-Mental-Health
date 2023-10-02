import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

data = pd.read_csv('./Student_mental_normalized2.csv')
# data.head()

columns_to_exclude = ['Age', 'Sex',
                      'Disability', 'Diagnosis', 'Diagnosis_text']

data_excluded = data.drop(columns_to_exclude, axis=1)
# data_excluded

correlation_matrix = data_excluded.corr()

# plt.figure(figsize=(50, 50))

# # Create a heatmap using seaborn
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# # Customize the plot
# plt.title('Correlation Heatmap')
# plt.show()

threshold = 0.33

columns_to_exclude = set()  # Set to store columns to be excluded

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            columns_to_exclude.add(colname)

columns_to_exclude = list(columns_to_exclude)

# Create a new DataFrame with the columns excluded
df_excluded = data_excluded.drop(columns=columns_to_exclude)

# Print the list of columns to be excluded
print("Columns to be excluded:", columns_to_exclude)

# Print the new DataFrame with columns excluded
print(df_excluded)

df_excluded.columns

df_excluded['Age'] = data['Age']
df_excluded['Sex'] = data['Sex']
df_excluded['Disability'] = data['Disability']
df_excluded['Diagnosis'] = data['Diagnosis']
df_excluded['Diagnosis_text'] = data['Diagnosis_text']
df_excluded['DASS_4'] = data['DASS_4']
df_excluded['DASS_6'] = data['DASS_6']
df_excluded['DASS_15'] = data['DASS_15']
df_excluded['DASS_21'] = data['DASS_21']
df_excluded['SSS_6'] = data['SSS_6']


columns_to_drop = ['Diagnosis', 'Diagnosis_text', 'DASS_2',
                   'Mindfulness_freq', 'Hobbies_Imp_5', 'Rested', 'SSS_2']

df_excluded = df_excluded[~(df_excluded['Diagnosis'] == 3)]

X = df_excluded.drop(columns=columns_to_drop, axis=1)
print(X.columns)
y = df_excluded['Diagnosis'] - 1

X_train, X_test, y_train, y_test = train_test_split(df_excluded.drop(columns=columns_to_drop, axis=1),
                                                    df_excluded['Diagnosis'], test_size=0.2,
                                                    random_state=101)


model = xgb.XGBClassifier()
model.fit(X_train, y_train-1)

predictions = model.predict(X_test)

print(accuracy_score(y_test-1, predictions))

k_fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(model, X, y, cv=k_fold, scoring='accuracy')

print(f'Cross-Validation Scores: {cross_val_scores}')
print(f'Average Accuracy: {cross_val_scores.mean():.2f}')
model_filename = 'xgboost_model.joblib'
joblib.dump(model, model_filename)
