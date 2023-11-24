import pandas as pd
import numpy as np
import helper

# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Combine train and test datasets
combined_data = pd.concat([train, test], axis=0)

# Fill Missing Values
missing_values_cols_numerical = [col for col in combined_data.columns if combined_data[col].isnull().any() and combined_data[col].dtype != 'object']
missing_values_cols_nonnumerical = [col for col in combined_data.columns if combined_data[col].isnull().any() and combined_data[col].dtype == 'object']

for col in missing_values_cols_numerical:
    combined_data[col].fillna(combined_data[col].median(), inplace=True)

for col in missing_values_cols_nonnumerical:
    combined_data[col].fillna('Missing', inplace=True) 

# Create Numerical Features
age = combined_data['Age']
siblings_spouses = combined_data['SibSp']
parents_children = combined_data['Parch']
fare = combined_data['Fare']

sex_mapping = {'male': 0, 'female': 1}
sex = combined_data['Sex'].map(sex_mapping)

cabin = combined_data['Cabin'].apply(helper.score_cabins)

embarked = pd.get_dummies(combined_data['Embarked'], prefix='Embarked')

# Combine all features
features = pd.concat([age, siblings_spouses, parents_children, fare, sex, cabin, embarked], axis=1)

# Split train and test datasets
features_train = features.iloc[:len(train)]
features_test = features.iloc[len(train):]

# Train Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(features_train, train['Survived'])

# Print score on training data
# print(model.score(features_train, train['Survived']))

# Predict On Test Data
predict = model.predict(features_test)

# write predictions to csv with column 1 as their PassengerIndex from test.csv
predict = pd.DataFrame(predict, columns=['Survived'])
predict.index += 892
predict.index.name = 'PassengerId'
predict.to_csv('predictions.csv')
