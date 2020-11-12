import numpy as np

class SimpleNeuralNetwork(object):
    def __init__(self, dataset, normalizer, theta_generator):
        self.__dataset = dataset
        self.__normalizer = normalizer
        self.__theta_generator = theta_generator

    @property
    def theta_generator(self):
        return self.__theta_generator

    @property
    def thetas(self):
        return self.__theta_generator.thetas

    def __setup_models(self):
        
        X = np.copy(self.__dataset)
        X[:, 0] = 1
        X[:, 1:] = self.__dataset[:, :-1]

        # X = self.__dataset[:, :-1]

        unique_values = np.unique(self.__dataset[:, -1])

        y = np.zeros((np.size(self.__dataset, 0), np.size(unique_values) if np.size(unique_values) > 2 else 1))

        if np.size(y, 1) == 1:
            true_positions = np.where(self.__dataset[:, -1] == unique_values[-1])[0]

            for position in range(np.size(y, 0)):
                y[position, 0] = 1 if position in true_positions else 0

        else:
            for position in range(np.size(y, 0)):
                y[position, np.where(unique_values == self.__dataset[position, -1])[0][0]] = 1

        return X, y

    @property
    def input_set(self):
        return self.__setup_models()[0]

    @property
    def output_values(self):
        return np.unique(self.__dataset[:, -1])

    @property
    def output_set(self):
        return self.__setup_models()[1]

    def predict(self, predict_set):
        return self.__theta_generator.predict(self.__normalizer.normalize_predict(predict_set))

    def learn(self):
        X, y = self.__setup_models()
        for information in self.__theta_generator.generate(self.__normalizer.normalize_input(X), y):
            yield information
