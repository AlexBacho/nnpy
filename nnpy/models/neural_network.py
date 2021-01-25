import pickle


class NeuralNetwork:
    def __init__(self, loss_function=None, prime_loss_function=None):
        self.layers = []
        self.loss_function = loss_function
        self.prime_loss_function = prime_loss_function

    def add_layers(self, *layers):
        self.layers.extend(layers)
        return self

    def use_loss_func(self, loss, prime_loss_function):
        self.loss_function = loss
        self.prime_loss_function = prime_loss_function
        return self

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            err = 0

            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss_function(y_train[j], output)

                error = self.prime_loss_function(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples
            print("epoch %d/%d   error=%f" % (i, epochs, err))
        return self

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def save(self, output_file):
        with open(output_file, "wb+") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def load_nn(input_file):
    with open(input_file, "rb") as f:
        return pickle.load(f)