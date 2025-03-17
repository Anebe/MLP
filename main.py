import math
import random
import load_data




def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    return 1 if x > 0 else alpha

random.seed(42)

MAX = 1
def hard_sigmoid(x):
    return max(0, min(1 * MAX, (0.2 * MAX) * x + (0.5 * MAX)))

def hard_sigmoid_derivative(x):
    if 0 < (0.2 * MAX) * x + (0.5 * MAX) < MAX:
        return (0.2 * MAX)
    else:
        return 0
    
def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


class Neuronio:
    def __init__(self, n_inputs, activation_func, activation_deriv_func):
        self.weights = [random.uniform(0, 0.7) for _ in range(n_inputs)]
        self.bias = random.uniform(0, 0.7)
        self.activation_func = activation_func
        self.activation_deriv_func = activation_deriv_func
        self.output = None
        self.error = None
        self.last_weighted_sum = None
        self.last_input = None
        self.MAX_GRAD = 10.0
    def multiply_arrays(self, a, b):
        if len(a) != len(b):
            raise ValueError("As listas devem ter o mesmo tamanho")
    
        return [x * y for x, y in zip(a, b)]
    

    def calculate_output(self, inputs):
        self.last_input = inputs
        
        multiply_arrays = self.multiply_arrays(inputs,self.weights) 
        sum_weights = sum(multiply_arrays)

        self.last_weighted_sum = sum_weights + self.bias
        self.output = self.activation_func(self.last_weighted_sum)
        return self.output

    def calculate_error_output_layer(self, y):
        self.error = (y[0]-self.output) * self.activation_deriv_func(self.last_weighted_sum + self.bias)
        self.error = max(min(self.error, self.MAX_GRAD), -self.MAX_GRAD)

        return self.error 
    
    def calculate_error_hidden_layer(self, error_next_layer, weights_next_layer):

        sum_error = 0
        for erro in error_next_layer:
            for neuro_wheigh in weights_next_layer:
                sum_error += erro * neuro_wheigh
        
        self.error = sum_error * self.activation_deriv_func(self.last_weighted_sum + self.bias)
        self.error = max(min(self.error, self.MAX_GRAD), -self.MAX_GRAD)
        return self.error

    def update_weights(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.error * self.last_input[i]
        self.bias -= learning_rate * self.error


class Layer:
    def __init__(self, n_neurons, n_inputs_per_neuron, activation_func, activation_deriv_func):
        self.neurons = [Neuronio(n_inputs_per_neuron, activation_func, activation_deriv_func) for _ in range(n_neurons)]

    def forward(self, inputs):
        result = []

        for neuron in self.neurons:
            result.append(neuron.calculate_output(inputs))
        return result

    def backward(self, error_next_layer=None, weights_next_layer=None, y_expected = None, learning_rate = 0):
        
        for i, neuron in enumerate(self.neurons):
            if y_expected:
                neuron.calculate_error_output_layer(y_expected)
                for j, _ in enumerate(neuron.weights):
                    neuron.weights[j] -= learning_rate * neuron.error * neuron.output
                neuron.bias -= neuron.error * learning_rate
                
            else:
                neuron.calculate_error_hidden_layer(error_next_layer, weights_next_layer[i])
                neuron.update_weights(learning_rate)
                
        error = [neuron.error for neuron in self.neurons]
        weights = [[0] * len(self.neurons) for _ in range(len(self.neurons[0].weights))]
        
        for i, neuro in enumerate(self.neurons):
            for j, weight in enumerate(neuro.weights):
                weights[j][i] = weight
                
        return error, weights


class MLP:
    def __init__(self, input_dim, layers_dims, output_dim, activation_func, activation_deriv_func):
        self.layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.activation_func = activation_func
        self.activation_deriv_func = activation_deriv_func
        
        prev_dim = input_dim
        for layer_dim in layers_dims:
            self.layers.append(Layer(layer_dim, prev_dim, activation_func, activation_deriv_func))
            prev_dim = layer_dim

        # Camada de saída
        self.output_layer = Layer(output_dim, prev_dim, activation_func, activation_deriv_func)
        self.layers.append(self.output_layer)

    def forward(self, X):
        inputs = X
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, y_result, y_expected, learning_rate):
        error_next_layer = None 
        weights_next_layer = None
        
        for layer in reversed(self.layers):
            error_next_layer, weights_next_layer = layer.backward(
                error_next_layer = error_next_layer,
                weights_next_layer = weights_next_layer,
                y_expected = y_expected,
                learning_rate = learning_rate)
            
            y_expected = None


    def train(self, X, y, epochs, learning_rate, on_end_epoch = None):
        for epoch in range(epochs):
            for i,j in zip(X,y):
                results = self.forward(i)
                self.backward(results, j, learning_rate)
            if on_end_epoch:
                on_end_epoch()

    def predict(self, X):
        return [self.forward(i) for i in X]




# Treinamento da rede neural
RANGE = 0.995
inputs, outputs = load_data.read_walmart()
inputs_train, inputs_test = load_data.split_array(inputs, RANGE)
outputs_train, outputs_test = load_data.split_array(outputs, RANGE)

mlp = MLP(input_dim=len(inputs_train[0]), 
          layers_dims=[2,3,4,2], 
          output_dim=len(outputs_train[0]),
          activation_func=leaky_relu,
          activation_deriv_func=leaky_relu_derivative)

def end_epoch():
    predictions = mlp.predict(inputs_test)
    for i, pred in enumerate(predictions):
        print(round(pred[0],2), outputs_test[i])
    print("---------------------")
        
# Treinamento
mlp.train(inputs_train, outputs_train, epochs=50, learning_rate=0.0001, 
          #on_end_epoch= end_epoch
          )
# Teste


end_epoch()
print("ERROR")
for i in mlp.layers:
    for j in i.neurons:
        print(j.error)
