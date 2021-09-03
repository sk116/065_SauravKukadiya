import numpy as np
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
targets = np.array([[56], [81], [119], [22], [103], 
                    [56], [81], [119], [22], [103], 
                    [56], [81], [119], [22], [103]], dtype='float32')
mu = np.mean(inputs, 0)
sigma = np.std(inputs, 0)
X = (inputs-mu) / sigma
X = np.hstack((np.ones((targets.size,1)),X))
print(X.shape)
rg = np.random.default_rng(12)
w = rg.random((1, 4))
print(w)
def mse(t1, t2):
    diff = t1 - t2
    return np.sum(diff * diff) / diff.size
def model(x,w):
    return x @ w.T
preds = model(X,w)
cost_initial = mse(preds, targets)
print("before regression cost is : ", cost_initial)
def gradient_descent(X, y, w, learning_rate, n_iters):
    history = np.zeros((n_iters, 1))
    for i in range(n_iters):
        h = model(X, w)
        diff = h - y
        delta = (learning_rate / targets.size) * (X.T@diff)
        new_w = w - delta.T
        w = new_w
        history[i] = mse(h, y)
    return (history, w)
import matplotlib.pyplot as plt
n_iters = 200
learning_rate = 0.01
initial_cost = mse(model(X, w),targets)
print("Initial cost is: ", initial_cost, "\n")
(history, optimal_params) = gradient_descent(X, targets, w, learning_rate, n_iters)
print("Optimal parameters are: \n", optimal_params, "\n")
print("Final cost is: ", history[-1])
import matplotlib.pyplot as plt
plt.plot(range(len(history)), history, 'g')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()
preds = model(X, optimal_params)
cost_final = mse(preds, targets)
print("Prediction:\n",preds)
print("Targets:\n", targets)
print("Cost after linear regression: ", cost_final)
print("Cost reduction : {} %".format(((cost_initial- cost_final) / cost_initial) * 100))
