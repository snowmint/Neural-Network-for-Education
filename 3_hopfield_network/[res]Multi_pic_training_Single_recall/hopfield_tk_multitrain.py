import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
# plot 顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
import hopfield_network as hopfield
from pathlib import Path
import subprocess
import glob, os, math

def flatten(t):
    return [item for sublist in t for item in sublist]

def read_data_txt(filename, list_get = []):
    with open(filename) as f:
        input_matrix = []
        for line in f:
            print("line:", line)
            input_line = []
            for ch in line:
                if ch == " ":
                    ch = "0"
                if ch != '\n':
                    input_line.append(ch)
            if input_line != []:
                #print("input_matrix", input_matrix)
                input_matrix.append(input_line)
            else:
                list_get.append(np.array(input_matrix, dtype='int'))
                input_matrix = []
                input_line = []
        list_get.append(np.array(input_matrix, dtype='int'))
        input_matrix = []
        input_line = []
    return list_get

def get_noise_input(input_data, noise_volume):
    # 加入隨機雜訊
    input_shape = input_data.shape
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if input_data[i][j] == 0:
                input_data[i][j] = -1
    noise_input = flatten(np.copy(input_data))
    bino = np.random.binomial(n = 1, p = noise_volume, size = len(noise_input))
    #print("bino", bino)
    for i, element in enumerate(noise_input):
        #print("i:", i, " | element:", element)
        if bino[i]: # 二項式分布選中的元素進行
            noise_input[i] = -1 * element
    return np.array(noise_input).reshape(input_shape), np.array(noise_input).reshape(input_shape)

class GUI_window:
    def __init__(self, win):
        x0, xt0, y0 = 50, 80, 80
        #---- zero label and entry -------
        self.label_file = tk.Label(win,text = "輸入檔案")
        self.label_file.config(font=('Arial', 10))
        self.label_file.grid(column=0, row=0)
        os.chdir("./Hopfield_dataset/")
        txt_file = []
        for file in glob.glob("*Train*.txt"):
            txt_file.append(file)
        self.label_file.place(x=x0, y=y0-10)
        self.Entry_file = ttk.Combobox(win,values=txt_file)
        self.Entry_file.grid(column=0, row=0)
        self.Entry_file.current(1)
        self.Entry_file.place(x=x0, y=y0+20)
        self.filename = str(self.Entry_file.get())
        print(self.filename)
        #---- First label and entry -------
        self.label_0 = tk.Label(win, text='step 上限')
        self.label_0.config(font=('Arial', 10))
        self.label_0.place(x=x0, y=y0 + 70)
        self.Entry_0 = tk.Entry()
        self.Entry_0.place(x=x0, y=y0 + 90)
        self.Entry_0.insert(tk.END, str(10))
        self.epoch = int(self.Entry_0.get())

         #---- Second label and entry -------
        self.label_1 = tk.Label(win, text='噪音程度(%)')
        self.label_1.config(font=('Arial', 10))
        self.label_1.place(x=x0, y=y0 + 130)
        self.Entry_1 = tk.Entry()
        self.Entry_1.place(x=x0, y=y0 + 150)
        self.Entry_1.insert(tk.END, str(10))
        self.noise_level = float(self.Entry_1.get())

        #---- Compute button -------
        self.btn1 = tk.Button(win, text='開始訓練', command= self.start)
        self.btn1.place(x=xt0, y=y0 + 250)

        self.figure = Figure(figsize=(4, 7))

        #---- Show the plot-------
        self.plots = FigureCanvasTkAgg(self.figure, win)
        self.plots.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=0)

        toolbar = NavigationToolbar2Tk(self.plots, win)
        toolbar.update()

    def start(self):
        self.filename = str(self.Entry_file.get())
        self.num_step = int(self.Entry_0.get())
        self.noise_level = float(self.Entry_1.get())

        if "Bonus" in self.filename:
            self.filename = "Bonus_Training.txt"
            self.filename_test = "Bonus_Testing.txt"
        else:
            self.filename = "Basic_Training.txt"
            self.filename_test = "Basic_Testing.txt"
        input_array = []
        input_array = read_data_txt(self.filename, input_array) #Bonus_Training #Basic_Training

        for i in range(len(input_array)):
            for j in range(len(input_array[i])):
                for k in range(len(input_array[i][j])):
                    if input_array[i][j][k] == 0:
                        input_array[i][j][k] = -1
        print(input_array)
        
        for count_i in range(len(input_array)):
            input_array = []
            input_array = read_data_txt(self.filename, input_array) #Bonus_Training #Basic_Training

            for i in range(len(input_array)):
                for j in range(len(input_array[i])):
                    for k in range(len(input_array[i][j])):
                        if input_array[i][j][k] == 0:
                            input_array[i][j][k] = -1
            print(input_array)
            input_shape = input_array[count_i].shape
            print(input_array[count_i].shape)

            input_shape = (input_shape[0],input_shape[1])
            model = hopfield.hopfield(input_shape)
            print('weights shape ', model.W.shape)

            #Training
            model.fit_all(input_array)
            # model.fit([input_array[count_i]])
            print(input_array,'訓練完成')
            messagebox.showinfo('訓練完成', '訓練完成，開始回想')

            test_array = []
            test_array = read_data_txt(self.filename_test, test_array) #Bonus_Testing #Basic_Testing
            print(test_array)

            #網路回想
            iteration = self.num_step
            output_async, energy_list_async, early_stop = model.predict(test_array[count_i], iteration, asyn = True)
            output_sync, energy_list_sync, early_stop = model.predict(test_array[count_i], iteration, asyn = False)

            self.figure.clear()
            self.figure.suptitle('Hopfield 網路學習與回想結果')
            axs0 = self.figure.add_subplot(321)
            axs0.set_title('訓練資料')
            axs0.imshow(input_array[count_i].reshape(input_shape)*255, cmap='binary')
            axs1 = self.figure.add_subplot(322)
            axs1.set_title('測試資料')
            axs1.imshow(test_array[count_i]*255, cmap='binary')
            axs2 = self.figure.add_subplot(323)
            axs2.set_title('非同步更新')
            axs2.imshow(output_async*255, cmap='binary')
            axs3 = self.figure.add_subplot(324)
            axs3.set_title('同步更新')
            axs3.imshow(output_sync*255, cmap='binary')
            axs4 = self.figure.add_subplot(3,1,3)
            axs4.plot(energy_list_async)
            axs4.plot(energy_list_sync)
            axs4.set_title('能量函數變化')
            axs4.annotate(int(energy_list_async[-1]), [len(energy_list_async)-1, energy_list_async[-1]])
            axs4.annotate(int(energy_list_sync[-1]), [len(energy_list_sync)-1, energy_list_sync[-1]])
            axs4.set_ylabel('Energy')
            axs4.set_xlabel('Step')
            axs4.legend(['Async','Sync'])
            axs4.grid(b = True, which = 'both')
            axs4.set_xlim(0, max(len(energy_list_async), len(energy_list_sync))-1)
            self.figure.tight_layout()
            self.plots.draw()
            messagebox.showinfo('回想結束', '回想結束')

        for count_i in range(len(input_array)):
            input_array = []
            input_array = read_data_txt(self.filename, input_array) #Bonus_Training #Basic_Training

            for i in range(len(input_array)):
                for j in range(len(input_array[i])):
                    for k in range(len(input_array[i][j])):
                        if input_array[i][j][k] == 0:
                            input_array[i][j][k] = -1
            print(input_array)
            input_shape = input_array[count_i].shape
            print(input_array[count_i].shape)
            keep_input_data = np.copy(input_array)

            input_shape = (input_shape[0],input_shape[1])
            model = hopfield.hopfield(input_shape)
            print('weights shape ', model.W.shape)

            #Training
            model.fit(input_array)
            print(input_array,'訓練完成')
            messagebox.showinfo('訓練完成', '訓練完成，開始回想')

            # test_array = []
            # test_array = read_data_txt(self.filename_test, test_array) #Bonus_Testing #Basic_Testing
            # print(test_array)

            plus_test = []
            test_array = []
            for data in keep_input_data:
                print("data:", data)
                test, noise_data = get_noise_input(data.reshape(input_shape), self.noise_level/100)
                plus_test.append(noise_data)
                test_array.append(test)

            print(test_array)

            #網路回想
            iteration = self.num_step
            output_async, energy_list_async, early_stop = model.predict(plus_test[count_i], iteration, asyn = True)
            output_sync, energy_list_sync, early_stop = model.predict(plus_test[count_i], iteration, asyn = False)

            self.figure.clear()
            self.figure.suptitle('Hopfield 網路學習與回想結果')
            axs0 = self.figure.add_subplot(321)
            axs0.set_title('訓練資料')
            axs0.imshow(input_array[count_i].reshape(input_shape)*255, cmap='binary')
            axs1 = self.figure.add_subplot(322)
            axs1.set_title('噪音測試資料')
            axs1.imshow(test_array[count_i]*255, cmap='binary')
            axs2 = self.figure.add_subplot(323)
            axs2.set_title('非同步更新')
            axs2.imshow(output_async*255, cmap='binary')
            axs3 = self.figure.add_subplot(324)
            axs3.set_title('同步更新')
            axs3.imshow(output_sync*255, cmap='binary')
            axs4 = self.figure.add_subplot(3,1,3)
            axs4.plot(energy_list_async)
            axs4.plot(energy_list_sync)
            axs4.set_title('能量函數變化')
            axs4.annotate(int(energy_list_async[-1]), [len(energy_list_async)-1, energy_list_async[-1]])
            axs4.annotate(int(energy_list_sync[-1]), [len(energy_list_sync)-1, energy_list_sync[-1]])
            axs4.set_ylabel('Energy')
            axs4.set_xlabel('Step')
            axs4.legend(['Async','Sync'])
            axs4.grid(b = True, which = 'both')
            axs4.set_xlim(0, max(len(energy_list_async), len(energy_list_sync))-1)
            self.figure.tight_layout()
            self.plots.draw()
            messagebox.showinfo('回想結束', '回想結束')

window = tk.Tk()
gui_window = GUI_window(window)
window.title('Hopfield')
window.geometry("650x600+10+10")
window.mainloop()
window.quit()
