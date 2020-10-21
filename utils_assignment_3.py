import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython import display
from sklearn.metrics import accuracy_score


def hardlim(x):
    return np.asarray(x > 0).astype(int)


class Neuron:
    def __init__(self, inputs, targets, bias=None, weights=None, learning_rate=1):

        self.inputs = inputs
        self.targets = targets
        self.n_inputs = inputs.shape[1]
        self.n_samples = inputs.shape[0]
        self.activation_function = hardlim
        self.learning_rate = learning_rate
        if weights is None:
            self.weights = np.random.normal(loc=0, scale=0.01,
                                            size=[self.n_inputs])
        else:
            self.weights = weights

        if bias is None:
            self.bias = np.random.normal(loc=0, scale=0.01)
        else:
            self.bias = bias
        self.weight_updates, self.bias_updates = [], []
        self.mse, self.accuracy = [], []
        self.accuracy.append(self.evaluate_accuracy(self.inputs, self.targets))

    def output(self, input):
        x = np.sum(input * self.weights) + self.bias
        return self.activation_function(x)

    def classify(self, inputs):
        y_pred = []
        for input in inputs:
            pred = self.output(input)
            y_pred.append(pred)
        return np.array(y_pred)

    def evaluate_accuracy(self, inputs, targets):
        y_pred = self.classify(inputs)
        return accuracy_score(targets, y_pred) * 100

    def train(self):
        for i in range(self.n_inputs):
            self.train_step(i)

    def train_step(self, i):
        pattern, target = self.inputs[i], self.targets[i]
        err = self.output(pattern) - target
        self.weights = self.weights - self.learning_rate * err * pattern
        self.bias = self.bias - self.learning_rate * err
        acc = self.evaluate_accuracy(self.inputs, self.targets)
        self.accuracy.append(acc)


class mlp_simulation:
    f, ax = plt.subplots(1, 2, figsize=[15, 5])
    display_handle = display.display(f, display_id='fig')

    def __init__(self, X, y, learning_rate=1):
        self.X = X
        self.y = y
        self.init_X = X  # in case init button is pressed 2nd time
        self.init_y = y
        self.learning_rate = learning_rate
        self.neuron = Neuron(self.X, self.y, learning_rate=self.learning_rate)

    def initialize_button(self, _):
        self.neuron = Neuron(self.init_X, self.init_y, learning_rate=self.learning_rate)
        self.plot_data_boundary_accuracy()

    def train_button(self, _):
        for j in range(self.neuron.n_samples):
            self.neuron.train_step(j)
            self.plot_data_boundary_accuracy(j=j)

    def start(self):
        button = widgets.Button(description="Initialize")
        display.display(button)
        button.on_click(self.initialize_button)

        button = widgets.Button(description="Train")
        display.display(button)
        button.on_click(self.train_button)

    def plot_data_boundary_accuracy(self, j=None):
        weights = self.neuron.weights
        bias = self.neuron.bias
        plot_line = lambda x: (-bias - weights[0] * x) / (weights[1] + 10e-5)

        self.ax[0].clear()
        self.ax[1].clear()

        self.ax[0].scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1], color='g', s=40)
        self.ax[0].scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], color='r', s=40)

        if j is not None:
            self.ax[0].scatter(self.X[j, 0], self.X[j, 1],
                               facecolor=sns.xkcd_rgb['bright yellow'],
                               edgecolor='k', s=100, lw=2)

        x1, x2 = self.X[:, 0].min() - 0.1, self.X[:, 0].max() + 0.1
        y1, y2 = self.X[:, 1].min() - 0.1, self.X[:, 1].max() + 0.1
        xx = np.arange(x1, x2, step=0.001)

        self.ax[0].plot(xx, plot_line(xx), c='k')
        self.ax[0].set_ylim([y1, y2])
        self.ax[0].set_xlim([x1, x2])

        epochs = [j + 1 for j in range(len(self.neuron.accuracy))]
        self.ax[1].scatter(epochs, self.neuron.accuracy, c='k')
        self.ax[1].plot(epochs, self.neuron.accuracy, c='k')
        self.ax[1].set_ylim([50, 100])
        self.ax[1].set_yticks(range(int(10 * (np.min(self.neuron.accuracy) // 10)), 110, 10))
        self.ax[1].set_xticks(epochs)

        self.ax[0].set_title('Decision boundary')
        self.ax[1].set_title('Training accuracy')

        self.display_handle.update(self.f)
