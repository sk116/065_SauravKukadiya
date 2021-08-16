import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns

datasets = pd.read_csv("Exercise-CarData.csv")

X = datasets.iloc[:, :-1].values

Y = datasets.iloc[:, -1].values

X_new = datasets.iloc[:, 1:3].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_new)

std = StandardScaler()
X_Std = std.fit_transform(X_new)

le = LabelEncoder()
X[ : ,0] = le.fit_transform(X[ : ,0])

datasets.dropna(how='all', inplace=True)

new_X = datasets.iloc[:, :-1].values
new_Y = datasets.iloc[:, -1].values
imputer = SimpleImputer(missing_values = np.nan, strategy="mean")
imputer = imputer.fit(new_X[:, 1:3])
new_X[:, 1:3] = imputer.transform(new_X[:, 1:3])

datasets.head()
datasets = datasets.iloc[:, :-1]
datasets.head()
corr = datasets.corr()
corr.head()
sns.heatmap(corr)
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
                
selected_columns = datasets.columns[columns]
selected_columns.shape
data = datasets[selected_columns]