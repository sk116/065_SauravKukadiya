import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_digits
main_data = load_digits()
plt.figure(figsize = (20, 20))
for i in range(32):
    plt.subplot(8, 8, i + 1)
    plt.imshow(main_data.images[i])
ohe = preprocessing.OneHotEncoder()
X_train, X_test, Y_train, Y_test = train_test_split(main_data.data, main_data.target, test_size = 0.35, random_state = 65)
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_predicted = gnb.predict(X_test)
print(f"Accuracy :- {metrics.accuracy_score(Y_test, Y_predicted)}")
main_data.images[3]
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, Y_predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')
print(f"Classification report for classifier {gnb}:\n"
      f"{metrics.classification_report(Y_test, Y_predicted)}\n")
disp = metrics.plot_confusion_matrix(gnb, X_test, Y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()

