# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
'''
DecisionTreeRegressor – regression version of decision trees
 
mean_absolute_error, mean_squared_error, r2_score – standard regression evaluation metrics
'''
 
# Create a dataset with continuous exam scores
data = {
    'Study_Hours': [1, 2, 3, 1, 5, 6, 1, 8, 2, 10],
    'Attendance': [60, 62, 65, 10, 72, 8, 85, 90, 9, 95],
    'Exam_Score': [40, 45, 50, 85, 60, 70, 75, 82, 88, 95]
}
 
df = pd.DataFrame(data)
print(df)
 
'''
This dataset simulates a real-world trend:
More study hours and higher attendance → higher exam score.
 
'''
 
 
# Independent variables/features [X] and target/output [Y]
X = df[['Study_Hours', 'Attendance']]
y = df['Exam_Score']
 
# 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Create and train the Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)
 
# The model learns to map relationships between study hours, attendance, and exam scores.
 
# Predict exam scores for the test set
y_pred = regressor.predict(X_test)
print("Predicted Exam Scores:", y_pred)
 
# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 
'''
MAE – average absolute error (lower = better)
 
MSE – penalizes larger errors (lower = better)
 
R² Score – how much variance in the target variable is explained by the model (1.0 = perfect)
'''
 
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
 
# Predict score for a new student
new_student = pd.DataFrame({'Study_Hours': [1.5], 'Attendance': [28]})
predicted_score = regressor.predict(new_student)
print(f"Predicted Exam Score for new student: {predicted_score[0]:.2f}")
 
'''
Key Takeaways:
Concept	      Classification	             Regression
Output	   Categorical (Pass/Fail)	      Continuous (Score 0–100)
Example_Model	DecisionTreeClassifier	  DecisionTreeRegressor
Evaluation	Accuracy, F1-score	           MAE, MSE, R²
Goal	Predict class label	             Predict numeric value
 
'''