import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
print(tf.__version__)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
targets = np.array([[56], [81], [119], [22], [103], 
                    [56], [81], [119], [22], [103], 
                    [56], [81], [119], [22], [103]], dtype='float32')
df_inputs = pd.DataFrame(inputs, columns = ['temp','rainfall','humidity'])
features = df_inputs.copy()
train_features = features[:10]
test_features = features[10:] 
df_targets = pd.DataFrame(targets, columns = ['apples'])
label = df_targets.copy()
train_label =  label[:10]
test_label = label[10:]
print(train_features)
print(train_label)
train_features.describe().transpose()[['mean', 'std']]
sns.pairplot(train_features[['temp', 'rainfall', 'humidity']], diag_kind='kde')
from sklearn.preprocessing import Normalizer
import sklearn.preprocessing
temp = np.array(train_features['temp'])
temp_normalizer = preprocessing.Normalization(input_shape=[1, ], axis = None)
temp_normalizer.adapt(temp)
temp_linear_model = tf.keras.Sequential([
    temp_normalizer,
    layers.Dense(units=1)           
])
temp_linear_model.summary()
temp_linear_model.predict(temp[1:6])
temp_linear_model.compile(
    optimizer = tf.optimizers.Adam(learning_rate = 0.1),
    loss='mean_absolute_error')
test_results = {}
test_results['temp_linear_model'] = temp_linear_model.evaluate(
    test_features['temp'],
    test_label, verbose=0)
test_results['temp_linear_model']
y = temp_linear_model.predict(test_features['temp'])
y
norm_test_features = np.linalg.norm(test_features['temp'])
norm_train_features = np.linalg.norm(train_features['temp'])
norm_train_label = np.linalg.norm(train_label['apples'])
normal_array_test_features = test_features['temp']/norm_test_features
normal_array_train_features = train_features['temp']/norm_train_features
normal_array_train_label = train_label['apples']/norm_train_label
print(normal_array_train_features)
print(normal_array_test_features)
print(normal_array_train_label)
def plot_Apples(x, y):
  plt.scatter(train_features['temp'], train_label['apples'], label = 'Data')
  plt.plot(x, y, color='g', label = 'Predictions')
  plt.xlabel('Temp')
  plt.ylabel('Apples')
  plt.legend()  
plot_Apples(test_features['temp'], y)
normalizer = preprocessing.Normalization(axis = -1)
normalizer.adapt(np.array(train_features))
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units = 1)
])
linear_model.predict(train_features[:9])
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate = 0.1),
    loss='mean_absolute_error')
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_label, verbose = 0)
test_results['linear_model']

