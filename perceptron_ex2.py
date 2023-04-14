import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Perceptron(object):
    
    def __init__(self, learning_rate=0.01, n_iter=50, shuffle=True):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle

    def _initialize_weights(self, m):
        self.w_ = np.zeros(m + 1)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = target - output
        self.w_ += self.learning_rate * xi.dot(error)
        cost = 0.5 * error ** 2
        return cost

    def _shuffle(self, X, y):
        seq = np.random.permutation(len(y))
        return X[seq], y[seq]

    def fit(self, X, y):

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return np.sign(self.net_input(X))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

n_samples = 200
st_dev = 0.4

x1_data = np.random.normal(2, st_dev, n_samples)
y1_data = np.random.normal(2, st_dev, n_samples)
g1_data = 1 * np.ones(n_samples)

x2_data = np.random.normal(4, st_dev, n_samples)
y2_data = np.random.normal(4, st_dev, n_samples)
g2_data = -1 * np.ones(n_samples)

x_data = np.concatenate((x1_data, x2_data))
y_data = np.concatenate((y1_data, y2_data))
g_data = np.concatenate((g1_data, g2_data)).astype(int)

data = np.array([x_data, y_data]).T

perceptron = Perceptron()

perceptron.fit(data, g_data)

grid_size = 0.01
grid = np.zeros((round(6 / grid_size), round(6 / grid_size)))

for i in range(0, round(6 / grid_size), 1):
  for j in range(0, round(6 / grid_size), 1):
    grid[i, j] = perceptron.activation([i * grid_size, j * grid_size])


print(perceptron.w_)

# grid = np.rot90(grid, 3)

# plt.plot(x1_data, y1_data, 'or', mfc='none')
# plt.plot(x2_data, y2_data, 'ob', mfc='none')

# cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
#     "", ["blue", "red"], 2)

# c = plt.imshow(grid.T, cmap=cmap, vmax=3, vmin=-3,
#                extent=[0, 6, 0, 6], interpolation='nearest', alpha=0.2)
# plt.colorbar(c)

# plt.show()

xx, yy = np.meshgrid(np.arange(0, 6, 0.01), np.arange(0, 6, 0.01))

fig1 = plt.figure(1)

w1 = 1
theta = -6

sep1 = np.arange(0, 6, 0.01)
sep2 = [- (w1 * point) - (theta) for point in sep1]

plt.plot(x1_data, y1_data, 'or', mfc='none')
plt.plot(x2_data, y2_data, 'ob', mfc='none')

plt.contour(xx, yy, grid.T)
plt.plot(sep1, sep2, '-', mfc='none')

fig1.show()


fig2 = plt.figure(2)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["blue", "red"], 2)

plt.plot(x1_data, y1_data, 'or', mfc='none')
plt.plot(x2_data, y2_data, 'ob', mfc='none')
plt.contour(xx, yy, grid.T)

fig2.show()

fig3, ax = plt.subplots(subplot_kw={"projection": "3d"})


ax.plot_surface(xx, yy, grid)

fig3.show()

input()
