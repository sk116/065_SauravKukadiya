import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from subprocess import call
from sklearn import preprocessing
main_data = pd.read_csv("Dataset2.csv")
label_encoder = preprocessing.LabelEncoder()
for data in main_data:
  print(f"\n\nHeading :- {data}")
  print(list(main_data[data]))
  main_data[data] = label_encoder.fit_transform(main_data[data])
  print(f"\n\nAfter the tranformation of {data}")
  print(list(main_data[data]))
combined_features = tuple(zip(main_data["Outlook"], main_data["Temp"], main_data["Wind"], main_data["Humidity"]))
print("After combined!")
print("Outlook, Temp, Wind, Humidity\n\n")
for pair in combined_features:
    print(pair)
main_data
x_train, x_test, y_train, y_test = train_test_split(combined_features, main_data["Class"], test_size = 0.1, random_state = 54)
print(x_train)
from sklearn import metrics
dtc = DecisionTreeClassifier(criterion = "entropy")
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
y_pred
y_test
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(x_test)
print("y predicted : ", y_pred)
print(f"Actual y_test {y_test}")
disp = metrics.plot_confusion_matrix(dtc, x_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
export_graphviz(dtc, out_file='tree_entropy.dot',
               feature_names=['outlook','temperature','humidity','wind'],
               class_names=['play_no','play_yes'], 
               filled=True)
call(['dot', '-Tpng', 'tree_entropy.dot', '-o', 'tree_entropy.png', '-Gdpi=600'])
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree_entropy.png'))
plt.axis('off');
plt.show();

