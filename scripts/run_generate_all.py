import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plot_3d_features import generate_all_triplets
import pandas as pd

df = pd.read_csv('student_lifestyle_dataset.csv')
created = generate_all_triplets(df, 'plots/knn')
print('created', len(created))
for p in created[:10]:
    print(p)
