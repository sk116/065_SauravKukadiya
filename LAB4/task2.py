import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from subprocess import call

from sklearn.datasets import load_breast_cancer
main_data = load_breast_cancer()
X, y = load_breast_cancer(return_X_y = True)

X
#feature_names
load_breast_cancer().feature_names

main_data

y

print("Features shape : ",X.shape)
print("Label shape: ",y.shape)

label = ['Benign','Malignant']

from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 65)

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))

y_pred = dtc.predict(X[17].reshape(1, -1))
print("Predicted : ",label[int(y_pred)])
print("Actual : ",label[y[17]])
export_graphviz(dtc, out_file='tree_entropy.dot',
               feature_names=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                              'mean smoothness', 'mean compactness', 'mean concavity',
                              'mean concave points', 'mean symmetry', 'mean fractal dimension',
                              'radius error', 'texture error', 'perimeter error', 'area error',
                              'smoothness error', 'compactness error', 'concavity error',
                              'concave points error', 'symmetry error',
                              'fractal dimension error', 'worst radius', 'worst texture',
                              'worst perimeter', 'worst area', 'worst smoothness',
                              'worst compactness', 'worst concavity', 'worst concave points',
                              'worst symmetry', 'worst fractal dimension'],
               class_names=['Benign','Malignant'], 
               filled=True)
call(['dot', '-Tpng', 'tree_entropy.dot', '-o', 'tree_entropy.png', '-Gdpi=600'])
plt.figure(figsize = (16, 20))
plt.imshow(plt.imread('tree_entropy.png'))
plt.axis('off');
plt.show();

