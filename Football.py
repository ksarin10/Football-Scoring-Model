#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:33:08 2023

@author: krishsarin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split




df = pd.read_csv('/Users/krishsarin/Downloads/archive/NFL Play by Play 2009-2018 (v5).csv')


df['scoring'] = df['field_goal_result'].notnull() | (df['touchdown'] == 1) | (df['extra_point_result'].notnull())


df['turnover'] = (df['fumble_lost'] == 1) | (df['interception'] == 1)


additional_columns = [
    'pass_location', 'air_yards', 'yards_after_catch', 'run_location', 'run_gap',
    'kick_distance', 'pass_length', 'home_team', 'away_team', 'posteam_type', 'defteam',
    'shotgun', 'no_huddle', 'total_home_score', 'total_away_score', 'posteam_score', 'defteam_score',

]


feature_columns = [
    'yardline_100', 'quarter_seconds_remaining', 'half_seconds_remaining', 'game_seconds_remaining',
    'game_half', 'quarter_end', 'drive', 'sp', 'qtr', 'down', 'goal_to_go', 'time', 'yrdln', 'ydstogo', 'ydsnet',
    'play_type', 'yards_gained', 'scoring', 'turnover'
]

selected_columns = [
    'home_team', 'away_team', 'shotgun', 'no_huddle', 'game_half', 'total_home_score', 'total_away_score',
    'qtr', 'play_type', 'quarter_seconds_remaining', 'half_seconds_remaining', 'scoring', 'turnover'
]

feature_columns.extend(additional_columns)
feature_columns = selected_columns


X_train = df[feature_columns]
y_train = df['scoring']  

numeric_columns = X_train.select_dtypes(include='number').columns
print(numeric_columns)


X_train_filled_numeric = X_train[numeric_columns].fillna(X_train[numeric_columns].mean())


categorical_columns = X_train.select_dtypes(exclude='number').columns


X_train_filled_categorical = X_train[categorical_columns].fillna('missing_category')
X_train_filled = pd.concat([X_train_filled_numeric, X_train_filled_categorical], axis=1)
print(X_train_filled)




X_train = pd.get_dummies(X_train, columns=['home_team', 'away_team', 'posteam_type', 'defteam', 'shotgun', 'no_huddle', 'play_type','time', 'yrdln', 'game_half'])
X_train = pd.get_dummies(X_train_filled)

print(X_train)


X_train['game_half'] = X_train['game_half'].apply(lambda x: 1 if '1' in x else 2)


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

for column in X_train_filled.columns:
    print(f"Column: {column}, dtype: {X_train[column].dtype}")

# Train the model
brf_model = BalancedRandomForestClassifier(random_state=42)
brf_model.fit(X_train, y_train)


y_pred_scoring = brf_model.predict(X_test)

predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_scoring
})


print(predictions_df)

true_predictions = predictions_df[predictions_df['Actual'] == True]

print(true_predictions)

accuracy_scoring = accuracy_score(y_test, y_pred_scoring)
print(f"Scoring Accuracy: {accuracy_scoring}")

precision_scoring = precision_score(y_test, y_pred_scoring, average='binary')  
print(f"Precision: {precision_scoring}")

recall_scoring = recall_score(y_test, y_pred_scoring, average='binary')
print(f"Recall: {recall_scoring}")


f1_scoring = f1_score(y_test, y_pred_scoring, average='binary')
print(f"F1 Score: {f1_scoring}")


conf_matrix_scoring = confusion_matrix(y_test, y_pred_scoring)
print(f"Confusion Matrix:\n{conf_matrix_scoring}")

print("Training Set Class Distribution:")
print(y_train.value_counts())

print("\nTesting Set Class Distribution:")
print(y_test.value_counts())

