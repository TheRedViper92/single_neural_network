from numpy import exp, random, dot, tanh

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.weight_matrix = 2 * random.random((3,1))-1

    #funkcja aktywacji
    def tanh(self, x):
        return tanh(x)

    #pochodna funkcji aktywacji
    def tanh_derivative(self, x):
        return 1.0 - tanh(x)**2

    #propagacja
    def forward_propagation(self, inputs):
        return self.tanh(dot(inputs, self.weight_matrix))

    #trening sieci
    def train(self, train_inputs, train_outputs, num_train_iterations):
        for iteration in range(num_train_iterations):
            output = self.forward_propagation(train_inputs)

            #obliczanie błędu
            error = train_outputs - output

            #obliczenie poprawki do wag
            adjustment = dot(train_inputs.T, error * self.tanh_derivative(output))

            #dostosowanie wag
            self.weight_matrix += adjustment
