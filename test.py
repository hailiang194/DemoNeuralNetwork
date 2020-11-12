from simple_neural_network import SimpleNeuralNetwork
from theta_generator import Backpropagation
from normalizer import NoNormalize
import numpy as np

def Test(dataset, theta_generator=None, normalizer=NoNormalize()):
    print('\n\nINFORMATION:')
    print('DATASET\n' + str(dataset))
    machine = SimpleNeuralNetwork(dataset, normalizer, theta_generator)
    print('Input set:\n' + str(machine.input_set))
    print('Output set:\n' + str(machine.output_set))
    print('Unique values: ' + str(machine.output_values))
    print('Pre-predict:\n' + str(machine.predict(machine.input_set)))
    print('Learning...')
    for time, error in machine.learn():
        if(time % 10000 == 0):
            print(str(time) + ": " + str(error))
    if np.size(machine.output_set, 1) == 1:
        # print('Predict:' + str(machine.predict(machine.input_set)))
        predict = machine.predict(machine.input_set)
        print('Predict:\n ' + str(predict))
        value = np.greater(predict, 0.5)
        print('Value:\n' + str(value))
    else:
        predict = machine.predict(machine.input_set)
        print('Predict:\n ' + str(predict))
        max_posibilities = (np.amax(predict, axis=1))
        print(max_posibilities)
        value = np.zeros(predict.shape)
        for index in range(np.size(value, 0)):
            # value[index, np.where(predict[index, :] == max_posibilities[index])[0][0]] = 1
            position = (np.where(predict[index, :] == max_posibilities[index])[0][0])
            value[index, position] = 1    

        print('Value:\n ' + str(value))
    input()

if __name__ == "__main__":
    Test(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]), theta_generator=Backpropagation(0.01, 60000, [3, 4, 1]))
    Test(np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]), theta_generator=Backpropagation(0.01, 60000, [3, 4, 1]))
    Test(np.array([[0, 1, 3], [5, 2, 1], [3, 8, 3], [1, 2, 2], [3, 5, 2], [2, 1, 1]]), Backpropagation(0.01, 60000, [3, 4, 3]))
    Test(np.array([[3, 5, 3], [5, 3, 1], [6, 2, 2], [4, 3, 4]]), Backpropagation(0.01, 60000, [3, 4, 4]))
    # Test(np.array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 0]]), Backpropagation(1, 60000, [3, 4, 1]))
