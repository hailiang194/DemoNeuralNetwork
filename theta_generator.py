import numpy as np

class ThetaGenerator(object):
    def __init__(self, num_each_layer):
        self._num_each_layer = num_each_layer
        self._thetas = []
        np.random.seed(1)
        for i in range(len(num_each_layer) - 1):
            self._thetas.append(np.random.random((num_each_layer[i], num_each_layer[i + 1])))

    def _predict_each_layer(self, predict_set):
        layers = [predict_set]
        for theta in self._thetas:
            h = layers[-1] @ theta
            # print(h)
            try:
                layers.append(1 / (1 + np.exp(-h)))
            except RuntimeWarning as identifier:
                print(h)

        return layers

    def predict(self, predict_set):
        return self._predict_each_layer(predict_set)[-1]

    def generate(self, input_set, output_set):
        pass

    def error_cost(self, input_set, output_set):
        return np.mean(np.abs(output_set - self.predict(input_set))) 

class Backpropagation(ThetaGenerator):
    def __init__(self, alpha, num_iterator, num_each_layer):
        super().__init__(num_each_layer)
        self.__alpha = alpha
        self.__num_iterator = num_iterator

    def __derivative_of_sigmoid(self, value):
        return value * (1 - value)

    def generate(self, input_set, output_set):
        for time in range(self.__num_iterator):

            layers = self._predict_each_layer(input_set)

            error = output_set - layers[-1]

            for index in reversed(range(1, len(self._num_each_layer))):
                delta = error * self.__derivative_of_sigmoid(layers[index])
                # print('Theta[%d]=%s' % (index - 1, self._thetas[index - 1]))

                self._thetas[index - 1] = self._thetas[index - 1] + self.__alpha * (layers[index - 1].T @ delta)

                error = delta @ self._thetas[index - 1].T


            yield (time, self.error_cost(input_set, output_set))
