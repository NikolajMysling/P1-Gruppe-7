import pandas as pd
from itertools import combinations

df = pd.read_csv('student_lifestyle_dataset.csv')
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if 'Student_ID' in numeric_cols:
    numeric_cols.remove('Student_ID')
print('numeric_cols =', numeric_cols)
trip = list(combinations(numeric_cols,3))
print('triplet count =', len(trip))
print('first 10 triplets =', trip[:10])
