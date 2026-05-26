# Import necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#Explanation:

#pandas – for handling datasets
#train_test_split – to divide data into training and testing sets
#DecisionTreeClassifier – our supervised ML algorithm
#metrics – to evaluate model performance

# Create a simple dataset


data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 3, 9],
    'Attendance': [60, 62, 65, 70, 72, 80, 85, 90, 92, 40, 20, 85],
    'Result': ['Pass', 'Pass', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail', 'Fail', 'Pass', 'Pass', 'Fail']
}



# Fractured dataset
###
#data = {
    ##'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 3, 9],
    ##'Attendance': [60, 62, 65, 70, 72, 80, 85, 90, 92, 60, 65, 90],
    ##'Result': ['Pass', 'Pass', 'Fail', 'Fail',
    #           'Pass', 'Fail', 'Pass', 'Fail',
    #           'Pass', 'Fail', 'Pass', 'Fail']
#}
###


df = pd.DataFrame(data)

print(df)

# We manually created a dataset where:

# Students with more study hours and higher attendance are more likely to pass.

# The target variable (label) is 'Result'.

# Define features (X) and target (y)
X = df[['Study_Hours', 'Attendance']]  # Independent/Input variables
y = df['Result']                       # Dependent variable (label)

# X → inputs used to predict

# y → actual output (Pass/Fail)

# Split dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#We split the dataset to train the model and then test it on unseen data to check performance.
#random_state ensures reproducibility.

# Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# This step involves supervised learning — the model learns the relationship between features (X_train) and labels (y_train).

# Predict the test set results
y_pred = model.predict(X_test)
print("Predicted Results:", y_pred.tolist())

#The model now predicts the outcomes for unseen data (the test set).

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Accuracy → how often predictions are correct
#Classification report → detailed view of precision, recall, F1-score

#Takeaways :

#Supervised learning means training with labeled data (inputs + known outputs).
#The model learns patterns and can then predict unseen outcomes.
#You can easily swap the model to LogisticRegression, RandomForestClassifier, or KNeighborsClassifier to compare results.

