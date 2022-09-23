import numpy as np
import math
import matplotlib.pyplot as plt
import random

def Binary(Vn):
    return 1.0 if (Vn > 0) else 0.0

def Sigmoid(Vn):
    S = 1 / (1 + np.exp(-Vn))
    return S

def random_split(input_data, num_to_divide):
    train_dataset = input_data.copy()
    #random.shuffle(train_dataset)
    test_dataset = []
    random_select = np.random.choice(int(len(input_data)), int(len(input_data)/num_to_divide), replace=False)
    for count_i in range(len(random_select)):
        test_dataset.append(list(input_data[int(random_select[count_i])]))

    train_dataset = np.delete(input_data, random_select, 0)

    print("Length of dataset: ", len(input_data))
    print("Length of train set: ", len(train_dataset))
    print("Length of test set: ", len(test_dataset))

    return np.array(train_dataset), np.array(test_dataset)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def dn_to_normalize(dn):
    dn_min, dn_max = min(dn), max(dn)
    for i, val in enumerate(dn):
        dn[i] = (val-dn_min) / (dn_max-dn_min)
        #print(dn[i])
    return dn

def flatten(t):
    return [item for sublist in t for item in sublist]

class single_perceptron():
    def __init__(self, input_data, threshold = -1, learning_rate = 0.7, dn_threshold = 0.9, re_init_weight = 0):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.input_data = input_data
        train_dataset, test_dataset = random_split(input_data, 3.0) #隨機分 2/3 為 training set， 1/3 為 testing set
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dn = []
        self.min_dn = 0
        self.max_dn = 0
        self.train_dn_normalized = []
        self.test_dn_normalized = []
        self.num_of_input = 0
        #print("input_data.shape[0]",input_data.shape[0])
        self.weights = np.zeros(input_data.shape[0])
        self.dn_threshold = float(dn_threshold)
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.best_weight = np.zeros(input_data.shape[0])
        self.max_train_accuracy = 0
        self.max_test_accuracy = 0
        self.max_train_accuracy_test = 0
        self.re_init_weight = re_init_weight

    def train(self, epochs):
        self.test_accuracy = 0
        self.train_accuracy = 0
        self.max_test_accuracy = 0
        self.max_train_accuracy = 0
        self.max_train_accuracy_test = 0

        Xn = self.train_dataset[:, [0, 1]] #輸入值
        print("輸入向量:\n", Xn)
        self.num_of_input = Xn.shape[1]

        total_dn = self.input_data[:, [2]]
        self.min_dn = min(total_dn)
        self.max_dn = max(total_dn)

        dn = self.train_dataset[:, [2]] #期望值
        #print("期望值", dn)
        dn = flatten(dn)
        #print("flatten", dn)
        if(self.min_dn != 0) or (self.max_dn != 1):
            self.train_dn_normalized = dn_to_normalize(dn)
            dn = self.train_dn_normalized
        else:
            self.train_dn_normalized = dn
        print("期望值向量:\n", dn)

        Wn = np.random.normal(size=Xn.shape[1]+1)
        Wn[0] = self.threshold #theta = threshold
        print("隨機初始化權重向量(鍵結值):\n", Wn)

        count = 0
        for ind_epoch in range(int(epochs)):
            count += 1
            if (self.re_init_weight != 0) and (count > self.re_init_weight): #重新初始化權重，避免找太久都找不到，或是卡在震盪
                Wn = np.random.normal(size=Wn.shape[0])
                Wn[0] = self.threshold #theta = threshold
            incorrect_count = 0
            for count_step in range(Xn.shape[0]):
                X = [-1] #bias
                #print("Xn shape", Xn.shape[0], " | ",Xn.shape[1], " | step:", count_step)
                for count_j in range(Xn.shape[1]):
                    X.append(Xn[count_step][count_j])

                # print("count_step", count_step)
                X = np.array(X)
                W_dot_X = np.dot(X, Wn)
                y = Binary(W_dot_X)
                if (y != dn[count_step]): #修正出新 W(n) = W(n-1) - eta*X(n-1)
                    incorrect_count += 1
                    eta_X = float(self.learning_rate) * X
                    #print("Wn",Wn)
                    #print("eta_X", eta_X)
                    Wn_next = Wn + ((dn[count_step] - y) * eta_X)
                    Wn = Wn_next
                    #print("\nNew Wn:\n", Wn)
                    #每次更新都畫線檢查
                    #x_tmp, y_tmp, coefficients_tmp = self.get_line(Wn)
                    #plt_tmp = self.draw_result_2d(x_tmp, y_tmp, coefficients_tmp)

            self.weights = Wn
            test_accuracy = self.test()
            #print("Xn shape", Xn.shape[0])
            #print("incorrect_count", incorrect_count)
            train_accuracy = float((Xn.shape[0]-incorrect_count) / Xn.shape[0])
            print(train_accuracy)
            if (train_accuracy > self.dn_threshold) and (train_accuracy > self.max_train_accuracy):
                self.max_train_accuracy = train_accuracy
                self.max_train_accuracy_test = test_accuracy
                self.best_weight = Wn
            if (incorrect_count == 0) and (test_accuracy > self.dn_threshold) and (train_accuracy > self.dn_threshold):
                print("early break!")
                break
        train_accuracy = float((Xn.shape[0]-incorrect_count) / Xn.shape[0])
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy

        if (self.train_accuracy < self.max_train_accuracy):
            print("切換為 train 最好的鍵結值")
            Wn = self.best_weight
            print(Wn)
            self.weights = Wn
            test_accuracy = self.test()
            self.train_accuracy = self.max_train_accuracy
            self.test_accuracy = test_accuracy

        print("train_accuracy: ", self.train_accuracy, " | test_accuracy: ", self.test_accuracy)

        return Wn, train_accuracy

    def test(self):
        Xn = self.test_dataset[:, [0, 1]] #test 輸入值
        dn = self.test_dataset[:, [2]] #期望值
        #print(dn)
        dn = flatten(dn)
        if(self.min_dn != 0) or (self.max_dn != 1):
            self.test_dn_normalized = dn_to_normalize(dn)
        else:
            self.test_dn_normalized = dn
        #print(dn)
        Wn = self.weights
        incorrect_count = 0
        for count_step in range(Xn.shape[0]):
            X = [-1] #bias
            for count_j in range(Xn.shape[1]):
                X.append(Xn[count_step][count_j])
            X = np.array(X)
            W_dot_X = np.dot(X, Wn)
            y = Binary(W_dot_X)
            if (y != dn[count_step]):
                incorrect_count += 1
        test_acc = float((Xn.shape[0]-incorrect_count) / Xn.shape[0])
        return test_acc

    def get_line(self, Wn):
        ans_X = np.array([-1, 1, 1],  dtype="float32")
        print("Wn[0]", Wn[0])
        first_item = (ans_X[0] * Wn[0]) * (-1)
        print("平移量", first_item)
        Bn = np.array([first_item])

        Wn_eliminate = np.array(Wn[1:])
        ans_X[1] = Bn[0]/Wn_eliminate[0]
        ans_X[2] = Bn[0]/Wn_eliminate[1]

        x = [ ans_X[1], 0 ]
        y = [ 0, ans_X[2] ]
        #print(x, y)

        coefficients = np.polyfit(x, y, 1)
        return x, y, coefficients

    def get_plane(self, Wn):
        ans_X = np.array([-1, 1, 1],  dtype="float32")
        first_item = (ans_X[0] * Wn[0]) * (-1)
        Bn = np.array([first_item])

        Wn_eliminate = np.array(Wn[1:])
        ans_X[1] = Bn[0]/Wn_eliminate[0]
        ans_X[2] = Bn[0]/Wn_eliminate[1]
        ans_X[3] = Bn[0]/Wn_eliminate[2]

        x = [ 0, 0, ans_X[1] ]
        y = [ 0, ans_X[2], 0 ]
        z = [ ans_X[3], 0, 0 ]

        coefficients = np.polyfit(x, y, z, 1)
        return x, y, coefficients

    def for_tk_draw(self):
        return self.train_dataset, self.test_dataset, self.train_accuracy, self.test_accuracy, self.train_dn_normalized, self.test_dn_normalized

    def draw_result_2d(self, x, y, coefficients):
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        fig, ax = plt.subplots()

        for index in range(train_dataset.shape[0]):
            #print(index)
            if (self.train_dn_normalized[index] == 0.0):
                ax.scatter(train_dataset[index][0], train_dataset[index][1], label="train-dataset d=0", c = "blue")
            elif (self.train_dn_normalized[index] == 1/3):
                ax.scatter(train_dataset[index][0], train_dataset[index][1], label="train-dataset d=1/3", c = "purple")
            elif (self.train_dn_normalized[index] == 1.0):
                ax.scatter(train_dataset[index][0], train_dataset[index][1], label="train-dataset d=1", c = "red")

        if (self.test_dn_normalized):
            for index in range(test_dataset.shape[0]):
                if (self.test_dn_normalized[index] == 0.0):
                    ax.scatter(test_dataset[index][0], test_dataset[index][1], label="test-dataset d=0", c = "green")
                elif (self.test_dn_normalized[index] == 1/3):
                    ax.scatter(test_dataset[index][0], test_dataset[index][1], label="test-dataset d=1/3", c = "pink")
                elif (self.test_dn_normalized[index] == 1.0):
                    ax.scatter(test_dataset[index][0], test_dataset[index][1], label="test-dataset d=1", c = "orange")

        polynomial = np.poly1d(coefficients)
        x_axis = np.linspace(-20,20,10)
        y_axis = polynomial(x_axis)
        ax.plot(x_axis, y_axis)
        ax.plot( x[0], y[0], 'yo' )
        ax.plot( x[1], y[1], 'yo' )
        plt.title("Trained Result")
        plt.suptitle(str("train_accuracy: " + str(self.train_accuracy) + "\ntest_accuracy: " + str(self.test_accuracy)), x=.90, y=.95, horizontalalignment='right', verticalalignment='top', fontsize = 8, c = "red")
        legend_without_duplicate_labels(ax)
        plt.grid('on')
        #plt.show()
        return plt
        #plt.show()
