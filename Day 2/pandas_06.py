import pandas as pd
 
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],

        'Age': [25, 30, 22, 35],

        'City': ['New York', 'London', 'Paris', 'Tokyo']}

df = pd.DataFrame(data)
 
print("Original DataFrame:")

print(df)
 
# Select the first row

first_row = df.iloc[0]

print("\nFirst row:")

print(first_row)
 
# Select the first two rows and the first column

subset = df.iloc[0:2, 0]

print("\nFirst two rows, first column:")

print(subset)
 
# Select specific rows and columns using lists of integers

specific_data = df.iloc[[0, 2], [1, 2]]

print("\nSpecific rows and columns:")

print(specific_data)
 
# Select all rows and the second column

second_column = df.iloc[:, 1]

print("\nSecond column:")

print(second_column)
