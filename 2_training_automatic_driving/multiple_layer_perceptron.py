import numpy as np
import sys

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def sigmoid_derivative(y):
    return np.multiply(y, (1.0 - y)) # phi'(Vj(n)) =  Yj(n)*(1-Yj(n))

class Layer:
    batch_size = None

    def __init__(self, n_units_current, n_units_next, bias, layer_id):
        self.layer_id = layer_id
        self.n_units_current = n_units_current
        self.n_units_next = n_units_next
        self.bias = bias

        self.summation = self.initialize_vector((self.n_units_current, Layer.batch_size))
        self.activation = self.initialize_vector((self.n_units_current, Layer.batch_size))
        self.set_activation()
        self.weight = self.initialize_weights()
        self.delta_error = self.initialize_vector((self.bias + self.n_units_current, Layer.batch_size))
        self.gradient_error = self.initialize_vector(self.weight.shape)
        self.gradient_approximation = self.initialize_vector(self.gradient_error.shape)

    def initialize_weights(self):
        if self.n_units_next is None:
            return np.array([])
        else:
            weights = np.random.randn(self.n_units_next * (self.bias + self.n_units_current))
            weights = weights.reshape(self.n_units_next, self.bias + self.n_units_current)
            return weights

    def initialize_vector(self, n_dimensions):
        return np.random.normal(size=n_dimensions)

    def set_activation(self):
        self.activation = sigmoid(self.summation)
        if self.bias:
            self.add_activation_bias()

    def add_activation_bias(self):
        if len(self.activation.shape) == 1:
            #[ 1 ] 設 bias = 1 加在陣列最上方
            #[act]
            self.activation = np.vstack((1, self.activation))
        else:
            #vertical stack up the self.activation
            #[ 1 ,  1 , ...,  1 ]
            #[act, act, ..., act]
            self.activation = np.vstack((np.ones(self.activation.shape[1]), self.activation))

    def get_derivative_of_activation(self):
        return sigmoid_derivative(self.activation)

    def update_weights(self, r):
        self.weight += -(r * self.gradient_error)

    def check_gradient_computation(self, atol):
        return np.allclose(self.gradient_error, self.gradient_approximation, atol=atol)

    def print_layer(self):
        print("weight:\n {} \n".format(self.weight))
        print("summation: {}".format(self.summation))
        print("activation: {}".format(self.activation))
        print("delta_error: {}".format(self.delta_error))
        print("gradient_error: {}".format(self.gradient_error))

#輸入層
class InputLayer(Layer):
    def __init__(self, n_units_current, n_units_next=None, bias=True, layer_id=0):
        Layer.__init__(self, n_units_current, n_units_next, bias, layer_id)
        self.summation = []


#隱藏層，必須要記錄目前的 neuron 和其下一個 neuron，以進行倒傳遞
class HiddenLayer(Layer):
    def __init__(self, n_units_current, n_units_next, bias=True, layer_id=None):
        Layer.__init__(self, n_units_current, n_units_next, bias, layer_id)


#輸出層，沒有 bias 也沒有 weight
class OutputLayer(Layer):
    def __init__(self, n_units_current, layer_id):
        Layer.__init__(self, n_units_current, n_units_next=None, bias=False, layer_id=None)
        self.gradient_error = []
        self.gradient_approximation = []


class Dest_LinearRegression(OutputLayer):
    def __init__(self, n_units_current, layer_id):
        OutputLayer.__init__(self, n_units_current, layer_id)

    def set_activation(self):
        self.activation = self.summation

    def get_derivative_of_activation(self):
        return np.ones(shape = self.activation.shape)

def net_constructer(layers_dim, batch_size):
    if len(layers_dim) < 2:
        print("Must have at least 2 layers")
        exit()

    Layer.batch_size = batch_size
    net = []
    # 第一階段: create input and hidden layers
    for i in np.arange(len(layers_dim) - 1, dtype=int):
        if i == 0:
            new_layer = InputLayer(layers_dim[i], layers_dim[i + 1], bias=True)
            net.append(new_layer)
        else:
            new_layer = HiddenLayer(layers_dim[i], layers_dim[i + 1], bias=True, layer_id=i)
            net.append(new_layer)

    # 建立輸出層
    new_layer = Dest_LinearRegression(layers_dim[-1], layer_id=len(layers_dim))
    net.append(new_layer)

    return net

class NeuralNet:
    def __init__(self,  layers_dim, batch_size, target):
        self.mse_function = eval('self.mean_squared_error')

        self.layers_dim = layers_dim
        self.layer_out_id = len(layers_dim) - 1
        self.batch_size = batch_size
        self.net = net_constructer(self.layers_dim, self.batch_size)

        self.data_X = None
        self.data_Y = None
        self.idx = None
        self.data_X_batch = None
        self.target_history = []

    def compute_gradient_approximation(self, i, weight_shift=1e-4):
        W_shape = self.net[i].weight.shape
        for j_w in np.arange(W_shape[1]):
            for i_w in np.arange(W_shape[0]):
                # shift to minus limit
                self.net[i].weight[i_w, j_w] = self.net[i].weight[i_w, j_w] - weight_shift
                shift_negative = self.mse_function()
                # remove shift
                self.net[i].weight[i_w, j_w] = self.net[i].weight[i_w, j_w] + weight_shift

                # shift to plus limit
                self.net[i].weight[i_w, j_w] = self.net[i].weight[i_w, j_w] + weight_shift
                shift_positive = self.mse_function()
                # remove shift
                self.net[i].weight[i_w, j_w] = self.net[i].weight[i_w, j_w] - weight_shift

                # approximate gradient
                self.net[i].gradient_approximation[i_w, j_w] = (shift_positive - shift_negative)/(2*weight_shift)

    def gradient_checking(self):
        for i in np.arange(0, self.layer_out_id, dtype=int)[::-1]:
            self.compute_gradient_approximation(i)
            if not self.net[i].check_gradient_computation(atol=1e-1):
                print("gradient_error:")
                print(self.net[i].gradient_error)
                print("\ngradient_approximation:")
                print(self.net[i].gradient_approximation)
                sys.exit("Error in compute gradient from layer " + str(self.net[i].layer_id))
        print("Gradient Checking is Matching!")

    def back_propagate_error(self):
        # 分為兩個階段
        #   1. Neuron 屬於 output layer ，並計算瞬間誤差
        #   2. Neuron 屬於 hidden layer ，並計算瞬間誤差

        # Output layer
        derv_cost_by_activation = -(self.data_Y[self.idx] - self.net[self.layer_out_id].activation)
        derv_activation_by_summation = self.net[self.layer_out_id].get_derivative_of_activation()
        self.net[self.layer_out_id].delta_error = np.multiply(derv_cost_by_activation, derv_activation_by_summation)

        # Hidden layers
        for i in np.arange(1, self.layer_out_id, dtype=int)[::-1]:
            d_next = self.net[i + 1].delta_error
            if self.net[i + 1].bias:
                d_next = d_next[1:]

            derv_summation_lnext_by_activation = self.net[i].weight.transpose().dot(d_next)
            derv_activation_by_summation = self.net[i].get_derivative_of_activation()
            self.net[i].delta_error = np.multiply(derv_summation_lnext_by_activation, derv_activation_by_summation)

    def compute_gradients_errors(self):
        # 更新每一層的誤差
        self.back_propagate_error()

        # Hidden layers
        for i in np.arange(0, self.layer_out_id, dtype=int)[::-1]:
            layer_cur_activations = self.net[i].activation
            layer_next_errors = self.net[i + 1].delta_error
            # If layer_next (l+1) has bias, remove its error row
            if self.net[i + 1].bias:
                layer_next_errors = layer_next_errors[1:]

            self.net[i].gradient_error = layer_next_errors.dot(layer_cur_activations.transpose())
            # Normalize the gradient by batch size
            self.net[i].gradient_error = self.net[i].gradient_error / self.batch_size

    def update_weights(self, r, check_grad):
        # 計算鍵結值的梯度誤差
        self.compute_gradients_errors()
        if check_grad:
            self.gradient_checking()

        for i in np.arange(0, self.layer_out_id, dtype=int)[::-1]:
            self.net[i].update_weights(r)

    def feed_forward(self):
        for i in np.arange(0, self.layer_out_id + 1, dtype=int):
            if i == 0:
                # 第一層從輸入獲得 input
                self.net[i].activation[1:] = self.data_X[self.idx, :].transpose()
            else:
                # 之後每層的 input 都是上一層的輸出
                self.net[i].summation = self.net[i - 1].weight.dot(self.net[i - 1].activation)
                self.net[i].set_activation()

    def train(self, X, Y, r, iterations, shuffle=False, check_grad=True):
        self.data_X = X
        self.data_Y = Y

        # 建立 MSE and Log-likelihood histories
        self.mse_history = []
        self.ll_history = []

        # 對輸入進行 shuffle
        data_X_ids_order = np.arange(self.data_X.shape[0], dtype=int)
        if shuffle:
            np.random.shuffle(data_X_ids_order)

        # 計算一個 epoch 的需要幾次 batch iterations
        itr_to_epoch = int(self.data_X.shape[0] / self.batch_size)
        if itr_to_epoch == 0:
            sys.exit("Batch 大小超過 input 大小")

        j = 0
        for i in np.arange(iterations):
            self.idx = data_X_ids_order[(j * self.batch_size):((j + 1) * self.batch_size)]
            # 計算 data 分批位置
            j = j + 1
            if j >= itr_to_epoch: # 每個 epoch 結束便 reset
                j = 0
            self.feed_forward()
            self.metric_record()
            self.update_weights(r, check_grad)
            if i >= 4: # Turn off gradient checking
                check_grad = False

        # 從最終的鍵結值計算最佳值
        self.metric_record()

    def predict(self, x_i):
        x_i = x_i.transpose()
        if len(x_i.shape) == 1:
            x_i = np.expand_dims(x_i, 1)

        n_predictions = x_i.shape[1]
        # Feed foward process
        for i in np.arange(0, self.layer_out_id + 1, dtype=int):
            if i == 0:
                self.net[i].activation[1:, :n_predictions] = x_i
            else:
                self.net[i].summation[:, :n_predictions] = self.net[i - 1].weight.dot(self.net[i - 1].activation[:, :n_predictions])
                self.net[i].set_activation()
        predictions = self.net[self.layer_out_id].activation[:, :n_predictions]
        return predictions

    def train_fitted(self):
        train_fitted = np.array(([]))
        for i in np.arange(self.data_X.shape[0], dtype=int):
            p = self.predict(self.data_X[i])
            train_fitted = np.append(train_fitted, p)
        return train_fitted

    def mean_squared_error(self):
        v = self.predict(self.data_X[self.idx])
        y = self.data_Y[self.idx]
        mse = 0.5 * np.power(y - v, 2)
        mse = np.sum(mse) / self.batch_size
        return mse

    def metric_record(self):
        target_value = self.mse_function()
        self.target_history.append(target_value)

    def get_history(self):
        return self.target_history
