import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# plot 顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def flatten(t):
    return [item for sublist in t for item in sublist]


class hopfield:
    def __init__(self, input_shape):
        self.train_data = []
        self.W = np.zeros([input_shape[0] * input_shape[1], input_shape[0] * input_shape[1]], dtype=np.int8)
    
    def fit_all(self, train_list):
        self.n = len(train_list)
        print("n = ", self.n)
        for count_list in range(self.n): #self.n
            train_list[count_list] = np.array(flatten(train_list[count_list]))
            self.p = len(train_list[count_list])
            print("p = ", self.p)
            for i in range(self.p):
                for j in range(i, self.p):
                    if i==j:
                        self.W[i][j] = 0
                    else:
                        w_ij = train_list[count_list][i] * train_list[count_list][j]
                        self.W[i][j] += w_ij
                        self.W[j][i] += w_ij
        self.minus_self = (self.n / self.p) * np.identity(self.p) #(n/p)I
        print("minus_self", self.minus_self)

        self.W = (1/self.p) * self.W - self.minus_self # W = (1/p) Σ x*x^T - (n/p)I
        self.complete_threshold = np.sum(self.W, axis=1).T
        print("self.complete_threshold:", self.complete_threshold)

    def fit(self, train_list):
        self.n = len(train_list)
        print("n = ", self.n)
        for count_list in range(self.n): #self.n
            train_list[count_list] = np.array(flatten(train_list[count_list]))
            self.p = len(train_list[count_list])
            print("p = ", self.p)
            for i in range(self.p):
                for j in range(i, self.p):
                    if i==j:
                        self.W[i][j] = 0
                    else:
                        w_ij = train_list[count_list][i] * train_list[count_list][j]
                        self.W[i][j] += w_ij
                        self.W[j][i] += w_ij
        self.minus_self = (self.n / self.p) * np.identity(self.p) #(n/p)I
        print("minus_self", self.minus_self)

        self.W = (1/self.p) * self.W - self.minus_self # W = (1/p) Σ x*x^T - (n/p)I
        self.complete_threshold = np.sum(self.W, axis=1).T
        print("self.complete_threshold:", self.complete_threshold)

    def sgn(self, old, u, n): #處理矩陣精度運算問題，取到小數點下五位
        if np.round(u, 5) > np.round(n, 5):
            return 1
        if np.round(u, 5) == np.round(n, 5):
            return old
        if np.round(u, 5) < np.round(n, 5):
            return -1

    def update(self,state,idx=None):
        if idx==None:
            #print("syn")
            new_state = np.matmul(self.W, state)
            #print("new_state", new_state.reshape(10,10))
            temp_state = new_state
            #temp_state = np.subtract(new_state, self.complete_threshold)
            #print("self.complete_threshold:", self.complete_threshold.reshape(10,10))
            #print("temp_state", temp_state.reshape(10,10))
            for i in range(len(new_state)):
                #print("old[i]:", state[i], " | new:[i]", new_state[i], " | theta[i]", self.complete_threshold[i])
                temp_state[i] = self.sgn(state[i], new_state[i], self.complete_threshold[i])
            state = temp_state
            #print("syn state:", state.reshape(10,10))
        else:
            #print("asyn")
            new_state = np.matmul(self.W[idx], state)
            #print(state[idx])
            #print(new_state)
            state[idx] = self.sgn(state[idx], new_state, self.complete_threshold[idx])
        return state

    def predict(self, input_pic, iteration, asyn=False, async_iteration=1):
        input_shape = input_pic.shape
        fig,axs = plt.subplots(1,1)
        axs.axis('off')
        print(input_shape)
        graph = axs.imshow(input_pic * 255, cmap='binary')
        input_pic = np.where(input_pic < 0.5,-1,1)
        fig.canvas.draw_idle()
        plt.pause(1)
        energy_list = []

        prev_energy = self.energy(input_pic.flatten())
        energy_list.append(prev_energy)
        state = input_pic.flatten()
        early_stop = iteration
        if asyn:
            for i in range(iteration):
                for j in range(async_iteration):
                    for index in range(self.p):
                        #print("idx", index)
                        state = self.update(state, index)
                        state_show = np.where(state < 1, 0, 1).reshape(input_shape)
                        graph.set_data(state_show * 255)
                        axs.set_title("非同步更新 step" + str(i) + " : " + str(index) + " pixel")
                        fig.canvas.draw_idle()
                        plt.pause(0.05) #0.25
                new_energy = self.energy(state)
                print('Step', i, ', Energy: ', new_energy)
                if new_energy == prev_energy:
                    print('能量函數未改變，停止更新')
                    early_stop = i
                    break
                prev_energy = new_energy
                energy_list.append(prev_energy)
        else:
            for i in range(iteration):
                state = self.update(state)
                state_show = np.where(state < 1, 0, 1).reshape(input_shape)
                graph.set_data(state_show * 255)
                axs.set_title("同步更新 step" + str(i))
                fig.canvas.draw_idle()
                plt.pause(0.5) #0.5
                new_energy = self.energy(state)
                print('Step', i, ', Energy: ', new_energy)
                if new_energy == prev_energy:
                    print('能量函數未改變，停止更新')
                    early_stop = i
                    break
                prev_energy = new_energy
                energy_list.append(prev_energy)
        plt.pause(1)
        plt.close()
        return np.where(state < 1,0,1).reshape(input_shape), energy_list, early_stop

    def energy(self, state): #能量若呈減少趨勢代表可穩定收歛
        obtain_energy = -(1.0/2.0) * np.matmul(np.matmul(state.T, self.W), state)
        return obtain_energy
