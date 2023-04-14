import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


# print(list(data.target_names))

class Perceptron(object):

    def __init__(self, learning_rate=0.0000001, n_iter=10, shuffle=True):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.err_arr = np.zeros(n_iter)

    def _initialize_weights(self, m):
        self.w_ = np.zeros(m)
        self.w_initialized = True

    def fit(self, X, y):

        self._initialize_weights(X.shape[1])

        ei2 = 0

        for epoch in range(self.n_iter):
            seq = np.random.permutation(len(y))

            for i in range(X.shape[1]):
                error = (y[seq[i]] - np.dot(self.w_.T, X[seq[i]]))
                self.w_ = self.w_ + self.learning_rate * error * X[seq[i]]
                ei2 = ei2 + error ** 2

            print(ei2)
            self.err_arr[epoch] = ei2

        return self

    def net_input(self, X):
        return np.dot(X, self.w_)

    def activation(self, X):
        return np.sign(self.net_input(X))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


dataset = load_breast_cancer()
x_data = dataset.data[:, : 10]
y_data = dataset.target[:]
# print(y_data)
y_data = np.array([round(np.sign(element - 0.5)) for element in y_data])

# print (y_data)

perceptron = Perceptron()

perceptron.fit(x_data, y_data)

# grid_size = 0.01
# grid = np.zeros((round(6 / grid_size), round(6 / grid_size)))

# for i in range(0, round(6 / grid_size), 1):
#   for j in range(0, round(6 / grid_size), 1):
#     grid[i, j] = perceptron.activation([i * grid_size, j * grid_size])


print(np.round(perceptron.activation(x_data)))
print(y_data)

# # grid = np.rot90(grid, 3)

# # plt.plot(x1_data, y1_data, 'or', mfc='none')
# # plt.plot(x2_data, y2_data, 'ob', mfc='none')

# # cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
# #     "", ["blue", "red"], 2)

# # c = plt.imshow(grid.T, cmap=cmap, vmax=3, vmin=-3,
# #                extent=[0, 6, 0, 6], interpolation='nearest', alpha=0.2)
# # plt.colorbar(c)

# # plt.show()

# xx, yy = np.meshgrid(np.arange(0, 6, 0.01), np.arange(0, 6, 0.01))

# fig1 = plt.figure(1)

# w1 = 1
# theta = -6

# sep1 = np.arange(0, 6, 0.01)
# sep2 = [- (w1 * point) - (theta) for point in sep1]

# plt.plot(x1_data, y1_data, 'or', mfc='none')
# plt.plot(x2_data, y2_data, 'ob', mfc='none')

# plt.contour(xx, yy, grid.T)
# plt.plot(sep1, sep2, '-', mfc='none')

# fig1.show()


# fig2 = plt.figure(2)

# cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
#     "", ["blue", "red"], 2)

# plt.plot(x1_data, y1_data, 'or', mfc='none')
# plt.plot(x2_data, y2_data, 'ob', mfc='none')
# plt.contour(xx, yy, grid.T)

# fig2.show()

# fig3, ax = plt.subplots(subplot_kw={"projection": "3d"})


# ax.plot_surface(xx, yy, grid)

# fig3.show()

# input()
