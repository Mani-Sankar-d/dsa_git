import numpy as np
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
# print(iris)
X = iris.data[["sepal length (cm)", "sepal width (cm)"]].values
y = iris['target'].values
X_with_bias = np.c_[np.ones(len(X)), X]
total_size = len(X_with_bias)
test_size = int(0.2*total_size)
val_size = int(0.2*total_size)
train_size = total_size-test_size-val_size
np.random.seed(42)
rnd_indices = np.random.permutation(total_size)
X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_val = X_with_bias[rnd_indices[train_size:-test_size]]
y_val = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

def to_one_hot(y):
    return np.diag(np.ones(y.max()+1))[y]
y_train_one_hot = to_one_hot(y_train)
y_val_one_hot = to_one_hot(y_val)
y_test_one_hot = to_one_hot(y_test)

#scale

mean = X_train[:, 1:].mean(axis=0)
std = X_train[:, 1:].std(axis=0)
X_train[:,1:] = (X_train[:,1:]-mean)/std
X_val[:,1:] = (X_val[:,1:]-mean)/std
X_test[:,1:] = (X_test[:,1:]-mean)/std


def softmax(logits):
    exps = np.exp(logits)
    exp_sums = exps.sum(axis=1, keepdims=True)
    return exps/exp_sums

n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))


n_epochs = 5001
m = len(X_train)
epsilon = 1e-5

np.random.seed(42)
theta = np.random.randn(n_inputs, n_outputs)

for epoch in range(n_epochs):
    logits = X_train@theta
    y_proba = softmax(logits)
    if(epoch%1000 == 0):
        Y_proba_val = softmax(X_val @ theta)
        xentropy_losses = -(y_val_one_hot * np.log(Y_proba_val + epsilon))
        l2_loss = 1/2 * (theta[1:]**2).sum()
        total_loss = xentropy_losses.sum(axis=1).mean()+0.01*l2_loss
        print(epoch, total_loss.round(4))
    error = y_proba-y_train_one_hot
    gradients = 1/m *X_train.T@error
    gradients += np.r_[np.zeros([1, n_outputs]), 0.01 * theta[1:]]
    theta = theta- 0.5*gradients
logits = X_val @ theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)

accuracy_score = (y_predict == y_val).mean()
print(accuracy_score)
