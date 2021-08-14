import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Data_for_Transformation.csv')
plt.scatter(data['Age'],data['Salary'])
plt.show()

plt.hist(data['Salary'],bins=5)
plt.show()

df = pd.DataFrame(data,columns=['Country'])
ans = df['Country'].value_counts()
ind = ans.index
vals = []
for x in ind:
    vals.append(ans[x])
plt.bar(ind,vals)
plt.show()

