import pandas as pd

data = {'Name': ['John', 'Alice', 'Bob', 'Alex', 'Rony'],
        'Age': [25.5, 30, 35, 40, 30],
        'City': ['New York', 'London', 'Paris', 'Dubai', 'SFO']}

#Type - Dictionary 
#type(data)

df = pd.DataFrame(data)
print (df)
#print (df.head)