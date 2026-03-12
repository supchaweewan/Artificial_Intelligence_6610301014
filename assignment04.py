import numpy as np
import matplotlib.pyplot as plt

def generate_spiral(n_samples, turns=2, noise=0.0):
    X = []
    y = []

    for c in [0, 1]:
        r = np.linspace(0, 1, n_samples)
        t = np.linspace(c*np.pi, c*np.pi + turns*2*np.pi, n_samples)
        t += np.random.randn(n_samples) * noise

        X.append(np.c_[r*np.sin(t), r*np.cos(t)])
        y.append(np.full(n_samples, c))

    return np.vstack(X), np.hstack(y).reshape(-1,1)


X_train, y_train = generate_spiral(200, turns=2)
X_test,  y_test  = generate_spiral(200, turns=4)


def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2


np.random.seed(42)

W1 = np.random.randn(2, 8)
b1 = np.zeros((1,8))

W2 = np.random.randn(8, 8)
b2 = np.zeros((1,8))

W3 = np.random.randn(8, 1)
b3 = np.zeros((1,1))
learning_rate = 0.03
epochs = 6000

for epoch in range(epochs):

    # ----- Forward -----
    z1 = X_train @ W1 + b1
    a1 = tanh(z1)

    z2 = a1 @ W2 + b2
    a2 = tanh(z2)

    z3 = a2 @ W3 + b3
    y_hat = tanh(z3)

    loss = np.mean((y_hat - y_train)**2)

    # ----- Backprop -----
    d3 = 2*(y_hat - y_train) * tanh_derivative(z3)
    dW3 = a2.T @ d3
    db3 = np.sum(d3, axis=0, keepdims=True)

    d2 = d3 @ W3.T * tanh_derivative(z2)
    dW2 = a1.T @ d2
    db2 = np.sum(d2, axis=0, keepdims=True)

    d1 = d2 @ W2.T * tanh_derivative(z1)
    dW1 = X_train.T @ d1
    db1 = np.sum(d1, axis=0, keepdims=True)

    # ----- Update -----
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss:.4f}")


def predict(X):
    a1 = tanh(X @ W1 + b1)
    a2 = tanh(a1 @ W2 + b2)
    return tanh(a2 @ W3 + b3)


train_loss = np.mean((predict(X_train) - y_train)**2)
test_loss  = np.mean((predict(X_test) - y_test)**2)

print("\nFinal Result")
print("Training loss :", train_loss)
print("Testing loss  :", test_loss)


def plot_decision_boundary(X, y, title):
    h = 0.02

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6,6))
    plt.contourf(xx, yy, Z > 0.5, alpha=0.35, cmap='coolwarm')
    plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='coolwarm', edgecolors='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


plot_decision_boundary(X_train, y_train, "19 Nodes - Training Data")
plot_decision_boundary(X_test,  y_test,  "19 Nodes - Testing Data")