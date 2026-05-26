# importing pandas package
import pandas as pd
 
# making data frame from csv file
data = pd.read_csv("data.csv", index_col ="Name")
 
# retrieving row by loc method
first = data.loc["D"]
second = data.loc["E"]
 
print(first, "\n\n\n", second)