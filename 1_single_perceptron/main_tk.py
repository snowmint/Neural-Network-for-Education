import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from single_perceptron import single_perceptron
import glob, os

def load_txt(filename):
    dataset = list()
    with open(filename, 'r') as file:
        txt_reader = reader(file)
        for row in txt_reader:
            if not row:
                continue
            row_split = row[0].split()
            #print(row_split)
            for count in range(len(row_split)):
                if (count == len(row_split)-1):
                    row_split[count] = float(row_split[count])
                else:
                    row_split[count] = float(row_split[count])
            dataset.append(row_split)
    return dataset

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

class GUI_window:
    def __init__(self, win):
        x0, xt0, y0 = 10, 100, 50
        #---- First label and entry -------
        self.label_file = tk.Label(win,text = "輸入檔案")
        self.label_file.grid(column=0, row=0)
        os.chdir("./")
        txt_file = []
        for file in glob.glob("*.txt"):
            txt_file.append(file)
        self.label_file.place(x=x0, y=y0)
        self.Entry_file = ttk.Combobox(win,values=txt_file)
        self.Entry_file.grid(column=0, row=0)
        self.Entry_file.current(1)
        self.Entry_file.place(x=xt0, y=y0)
        self.filename = str(self.Entry_file.get())
        print(self.filename)
        #---- First label and entry -------
        self.label_0 = tk.Label(win, text='學習率(0~1)')
        self.label_0.config(font=('Arial', 10))
        self.label_0.place(x=x0, y=y0 + 40)
        self.Entry_0 = tk.Entry()
        self.Entry_0.place(x=xt0, y=y0 + 40)
        self.Entry_0.insert(tk.END, str(0.7))
        self.learning_rate = float(self.Entry_0.get())

        #---- Second label and entry -------
        self.label_1 = tk.Label(win, text='epochs')
        self.label_1.config(font=('Arial', 10))
        self.label_1.place(x=x0, y=y0 + 80)
        self.Entry_1 = tk.Entry()
        self.Entry_1.place(x=xt0, y=y0 + 80)
        self.Entry_1.insert(tk.END, str(300))
        self.epochs = float(self.Entry_1.get())

        #---- Third label and entry -------
        self.label_2 = tk.Label(win, text='收斂條件:期望 test 和 train 的準確率(0~1)')
        self.label_2.config(font=('Arial', 10))
        self.label_2.place(x=x0, y=y0 + 120)
        self.Entry_2 = tk.Entry()
        self.Entry_2.place(x=xt0, y=y0 + 140)
        self.Entry_2.insert(tk.END, str(0.9))
        self.dn_threshold = float(self.Entry_2.get())

        #---- Third label and entry -------
        self.label_3 = tk.Label(win, text='經過多少 epoch 要初始化權重(不要可填 0)')
        self.label_3.config(font=('Arial', 10))
        self.label_3.place(x=x0, y=y0 + 180)
        self.Entry_3 = tk.Entry()
        self.Entry_3.place(x=xt0, y=y0 + 200)
        self.Entry_3.insert(tk.END, str(100))
        self.re_init_weight = float(self.Entry_3.get())

        #---- Compute button -------
        self.btn = tk.Button(win, text='開始訓練', command= self.start)
        self.btn.place(x=xt0, y=y0 + 250)

        self.figure = Figure(figsize=(5, 10), dpi=100)
        self.subplot1 = self.figure.add_subplot(111)

        #---- Show the plot-------
        self.plots = FigureCanvasTkAgg(self.figure, win)
        self.plots.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=0)

        toolbar = NavigationToolbar2Tk(self.plots, win)
        toolbar.update()

    def start(self):
        self.learning_rate = float(self.Entry_0.get())
        self.epochs = float(self.Entry_1.get())
        self.dn_threshold = float(self.Entry_2.get())
        self.re_init_weight = float(self.Entry_3.get())

        self.filename = str(self.Entry_file.get())
        input_data = np.loadtxt( self.filename, dtype=float)
        #print(input_data)
        perceptron = single_perceptron(input_data, -1, self.learning_rate, self.dn_threshold, self.re_init_weight)
        Wn, train_accuracy = perceptron.train(self.epochs)
        if (input_data.shape[1] == 3):
            self.subplot1.clear()
            x, y, coefficients = perceptron.get_line(Wn)
            train_dataset, test_dataset, train_accuracy, test_accuracy, train_dn_normalized, test_dn_normalized = perceptron.for_tk_draw()
            return_plt = perceptron.draw_result_2d(x, y, coefficients)
            self.draw(Wn, train_dataset, test_dataset, train_accuracy, test_accuracy, x, y, coefficients, train_dn_normalized, test_dn_normalized) #call draw function
        elif (input_data.shape[1] == 4):
            x, y, z, coefficients = perceptron.get_plane(Wn)
            perceptron.draw_result_3d(x, y, z, coefficients)
        else:
            print("Can't draw the picture over 3D space")

    def draw(self, Wn, train_dataset, test_dataset, train_accuracy, test_accuracy, x, y, coefficients, train_dn_normalized, test_dn_normalized):
        for index in range(train_dataset.shape[0]):
            if (train_dn_normalized[index] == 0.0):
                self.subplot1.scatter(train_dataset[index][0], train_dataset[index][1], label="train-dataset d=0", c = "blue")
            elif (train_dn_normalized[index] == 1/3):
                self.subplot1.scatter(train_dataset[index][0], train_dataset[index][1], label="train-dataset d=1/3", c = "purple")
            elif (train_dn_normalized[index] == 1.0):
                self.subplot1.scatter(train_dataset[index][0], train_dataset[index][1], label="train-dataset d=1", c = "red")
        for index in range(test_dataset.shape[0]):
            if (test_dn_normalized[index] == 0.0):
                self.subplot1.scatter(test_dataset[index][0], test_dataset[index][1], label="test-dataset d=0", c = "green")
            elif (test_dn_normalized[index] == 1/3):
                self.subplot1.scatter(test_dataset[index][0], test_dataset[index][1], label="test-dataset d=1/3", c = "pink")
            elif (test_dn_normalized[index] == 1.0):
                self.subplot1.scatter(test_dataset[index][0], test_dataset[index][1], label="test-dataset d=1", c = "orange")

        polynomial = np.poly1d(coefficients)
        x_axis = np.linspace(-20,20,10)
        y_axis = polynomial(x_axis)
        self.subplot1.plot(x_axis, y_axis)
        self.subplot1.plot( x[0], y[0], 'yo' )
        self.subplot1.plot( x[1], y[1], 'yo' )
        self.subplot1.set_title("Trained Result")
        legend_without_duplicate_labels(self.subplot1)
        self.subplot1.grid('on')
        self.plots.draw()
        messagebox.showinfo('Accuracy result', str("訓練準確率: " + str(train_accuracy) + "\n測試準確率: " + str(test_accuracy) + "\n鍵結值: " + str(Wn)))

window = tk.Tk()
gui_window = GUI_window(window)
window.title('Perceptron')
window.geometry("800x400+10+10")
window.mainloop()
window.quit()
