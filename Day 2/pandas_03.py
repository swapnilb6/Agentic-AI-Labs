# Pandas DataFrame Using Python List
import pandas as pd

# create a two-dimensional list
data = [['John', 25, 'New York', 200],
        ['Alice', 30, 'London', 300],
        ['Bob', 35, 'Paris', 400]]


# create a DataFrame from the list
df = pd.DataFrame(data, columns=['Name', 'Age', 'City', 'Salary'])

 

print(df)