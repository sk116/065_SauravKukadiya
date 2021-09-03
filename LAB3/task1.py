from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
import pandas as pd
data = pd.read_csv('Dataset2.csv')
outlook = data['Outlook']
temp = data['Temp']
wind = data['Wind']
humidity = data['Humidity']
label = data['Class']
le = preprocessing.LabelEncoder()
for d in data:
  print(f"\n\nHeading :- {data}")
  print(list(data[d]))
  data[d] = le.fit_transform(data[d])
  print(f"\n\nAfter the tranformation of {data}")
  print(list(data[d]))
outlook_encode = le.fit_transform(outlook)
temp_encoded = le.fit_transform(temp)
wind_encoded = le.fit_transform(wind)
humidity_encoded = le.fit_transform(humidity)
label_encoded = le.fit_transform(label)
features = tuple(zip(outlook_encode, temp_encoded, wind_encoded, humidity_encoded))
print("After combined;")
print("Outlook, Temp, Wind, Humidity\n\n")
for tpl in features:
    print(tpl)
X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size = 0.10, random_state = 65)
model = MultinomialNB()
model.fit(X_train, Y_train)
Y_predicted = model.predict(X_test)
print(Y_predicted)
print(f"Accuracy is :- {metrics.accuracy_score(Y_test, Y_predicted)}")
precision = precision_score(Y_test, Y_predicted)
recall = recall_score(Y_test, Y_predicted)
print(f"precision :- {precision}")
print(f"recall :- {recall}")
output = model.predict([[0, 1, 1, 2]])
print(f"Final prediction :- {output}")
output = model.predict([[2, 1, 0, 1]])
print(f"Final prediction :- {output}")
