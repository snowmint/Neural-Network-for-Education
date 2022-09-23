import time
import datetime
import glob, os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from itertools import cycle
#####################################
from sklearn.preprocessing import MinMaxScaler
import multiple_layer_perceptron as mlp
#from multiple_layer_perceptron import *
#####################################
from pathlib import Path
import subprocess
import sys

my_dir = Path(__file__).parent

def rotate(x, y, phi, center_x, center_y):
    new_x = math.cos(phi) * x - math.sin(phi) * y + center_x
    new_y = math.cos(phi) * y + math.sin(phi) * x + center_y
    return new_x, new_y

def is_on_line(x, y, x1, y1, x2, y2):
    #交點有沒有在牆壁線段上
    intersection_with_line1_a = (x-x1)**2 + (y-y1)**2
    intersection_with_line1_b = (x-x2)**2 + (y-y2)**2
    intersection_with_line1_o = (x1-x2)**2 + (y1-y2)**2
    # print("(x, y) = (", x, ", ", y, ")")

    if(intersection_with_line1_a + intersection_with_line1_b > intersection_with_line1_o):
        return False
    return True

def arrive_goal(point, goal1, goal2):
    v1_x = goal2[0] - goal1[0]
    v1_y = goal2[1] - goal1[1]

    v2_x = point[0] - goal1[0]
    v2_y = point[1] - goal1[1]

    if v1_x * v2_y - v1_y * v2_x >= 0:
        return True
    return False

def get_distance(line, point):
    x1, y1 = line[0][0], line[0][1]
    x2, y2 = line[1][0], line[1][1]

    x = point[0]
    y = point[1]

    d = 0
    if x1 != x2:
        m = 0
        if x1 - x2 != 0:
            m = (y1 - y2) / (x1 - x2)
        b = ((y1 + y2) - m * (x1 + x2)) / 2

        d = abs(m * x - y + b) / math.sqrt(m**2 + 1)
    else:
        d = abs(x - y + x1) / math.sqrt(1**2 + 1)

    return d

def get_intersection(line1, line2):
    x1, y1 = line1[0][0], line1[0][1]
    x2, y2 = line1[1][0], line1[1][1]

    x3, y3 = line2[0][0], line2[0][1]
    x4, y4 = line2[1][0], line2[1][1]

    if ((x2 - x1) == 0) and ((x4 - x3) == 0) : # 平行於 y 軸
        return None
    elif ((x2 - x1) == 0): # 一定與 y 軸平行的線段有交點
        x_know = x2
        k_test = (y4 - y3) / (x4 - x3)
        b_test = y3 - k_test * x3
        #y = kx +b
        y = k_test * x_know + b_test
        if is_on_line(x_know, y, x1, y1, x2, y2):
            return [x_know, y]
        else:
            return None

    elif ((x4 - x3) == 0):
        x_know = x4
        k_test = (y2 - y1) / (x2 - x1)
        b_test = y1 - k_test * x1
        #y = kx +b
        y = k_test * x_know + b_test
        if is_on_line(x_know, y, x1, y1, x2, y2):
            return [x_know, y]
        else:
            return None

    k = (y2 - y1) * 1.0 / (x2 - x1)
    b = y1 * 1.0 - x1 * k * 1.0
    if ((x4-x3) == 0):
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b) * 1.0 / (k - k2)

    y = (k * x * 1.0) + (b * 1.0)
    if is_on_line(x, y, x1, y1, x2, y2):
        return [x, y]
    else:
        return None

def closePlots(plt):
    plt.clf()
    plt.cla()
    plt.close("all")
    time.sleep(0.5)

def draw_wall(rail_map, circle_center, input_phi):
    # 初始 (x, y, Phi degree)
    # 終點 ['18,40'], ['30,37']
    x = rail_map[0][0]
    y = rail_map[0][1]
    # phi = rail_map[0][2]
    # print("initial: ( ", x, ", ", y, ", ",input_phi, ")")

    final_1 = rail_map[1]
    final_2 = rail_map[2]

    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(111)

    wall_line = []
    for i in range(3, len(rail_map)-1):
        # print(rail_map[i], rail_map[i+1])
        wall_line.append([rail_map[i], rail_map[i+1]])
        transpose_map = np.array([rail_map[i], rail_map[i+1]]).T.tolist()
        #print(transpose_map)
        plt.plot(transpose_map[0], transpose_map[1], color='b')
        plt.scatter(transpose_map[0], transpose_map[1], color='b')

    transpose_map = np.array([final_1, final_2]).T.tolist()
    plt.plot(transpose_map[0], transpose_map[1], color='r')
    plt.scatter(transpose_map[0], transpose_map[1], color='b')

    return wall_line, plt, ax

def find_bisector(point1, point2):
    mid_point = []
    mid_point[0] = (point1[0]+point2[0]) / 2.0
    mid_point[1] = (point1[1]+point2[1]) / 2.0

def draw_intersection(wall_line, circle_center, phi):
    line_equ = []
    distance_front_list = []
    distance_left_list = []
    distance_right_list = []

    for point in wall_line:
        # print(point[0]," | ", point[1])
        # print(phi)
        x1, y1 = rotate(4, 0, math.radians(phi), circle_center[0], circle_center[1])
        x_left, y_left = rotate(4, 0, math.radians(phi+45), circle_center[0], circle_center[1])
        x_right, y_right = rotate(4, 0, math.radians(phi-45), circle_center[0], circle_center[1])

        # print(circle_center, " | ",  [x1, y1])
        intersect_point_front = get_intersection((point[0], point[1]), (circle_center, [x1, y1]))
        #find bisector line of (circle_center, [x1, y1])
        # if intersection point 代入 bisector < 0 => 反向,其中垂線正負由 circle_center < 0 決定
        #末尾 - 頭
        if intersect_point_front != None:
            # print("intersect_point_front", intersect_point_front)
            vector_A = []
            vector_A.append(intersect_point_front[0] - circle_center[0])
            vector_A.append(intersect_point_front[1] - circle_center[1])
            vector_B = []
            vector_B.append(x1 - circle_center[0])
            vector_B.append(y1 - circle_center[1])
            dot_of_line = np.array(vector_A).dot(np.array(vector_B))

            # print(dot_of_line)
            if(dot_of_line > 0):#銳角
                plt.scatter(intersect_point_front[0], intersect_point_front[1], color='pink')
                dis_front = math.sqrt((circle_center[0]-intersect_point_front[0])**2 + (circle_center[1]-intersect_point_front[1])**2)
                distance_front_list.append(dis_front)
                # print("intersect_point_front: ", intersect_point_front)
        intersect_point_left = get_intersection((point[0], point[1]), (circle_center, [x_left, y_left]))
        if intersect_point_left != None:
            vector_A = []
            vector_A.append(intersect_point_left[0] - circle_center[0])
            vector_A.append(intersect_point_left[1] - circle_center[1])
            vector_B = []
            vector_B.append(x1 - circle_center[0])
            vector_B.append(y1 - circle_center[1])
            dot_of_line = np.array(vector_A).dot(np.array(vector_B))

            # print(dot_of_line)
            if(dot_of_line > 0):#銳角
                plt.scatter(intersect_point_left[0], intersect_point_left[1], color='lawngreen')
                dis_left = math.sqrt((circle_center[0]-intersect_point_left[0])**2 + (circle_center[1]-intersect_point_left[1])**2)
                distance_left_list.append(dis_left)
                # print("intersect_point_left: ", intersect_point_left)
        intersect_point_right = get_intersection((point[0], point[1]), (circle_center, [x_right, y_right]))
        if intersect_point_right != None:
            # print("intersect_point_right", intersect_point_right)
            vector_A = []
            vector_A.append(intersect_point_right[0] - circle_center[0])
            vector_A.append(intersect_point_right[1] - circle_center[1])
            vector_B = []
            vector_B.append(x1 - circle_center[0])
            vector_B.append(y1 - circle_center[1])
            dot_of_line = np.array(vector_A).dot(np.array(vector_B))

            # print(dot_of_line)
            if(dot_of_line > 0):#銳角
                plt.scatter(intersect_point_right[0], intersect_point_right[1], color='cyan')
                dis_right = math.sqrt((circle_center[0]-intersect_point_right[0])**2 + (circle_center[1]-intersect_point_right[1])**2)
                distance_right_list.append(dis_right)
                # print("intersect_point_right: ", intersect_point_right)

    distance_front_list = sorted(distance_front_list)
    distance_left_list = sorted(distance_left_list)
    distance_right_list = sorted(distance_right_list)
    # print("dis_front: ",distance_front_list)
    # print("dis_left: ",distance_left_list)
    # print("dis_right: ",distance_right_list)

    return distance_front_list, distance_left_list, distance_right_list

# Create labels
def ground_prediction(x1_i, x2_i):
        return x1_i + x2_i

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

class GUI_window:
    def __init__(self, win):
        x0, xt0, y0 = 10, 100, 50
        #---- zero label and entry -------
        self.label_file = tk.Label(win,text = "輸入檔案")
        self.label_file.config(font=('Arial', 10))
        self.label_file.grid(column=0, row=0)
        os.chdir("./")
        txt_file = []
        for file in glob.glob("train*.txt"):
            txt_file.append(file)
        self.label_file.place(x=x0, y=y0-10)
        self.Entry_file = ttk.Combobox(win,values=txt_file)
        self.Entry_file.grid(column=0, row=0)
        self.Entry_file.current(1)
        self.Entry_file.place(x=xt0, y=y0-10)
        self.filename = str(self.Entry_file.get())
        print(self.filename)
        #---- First label and entry -------
        self.label_0 = tk.Label(win, text='學習率(0~1)')
        self.label_0.config(font=('Arial', 10))
        self.label_0.place(x=x0, y=y0 + 20)
        self.Entry_0 = tk.Entry()
        self.Entry_0.place(x=xt0, y=y0 + 20)
        self.Entry_0.insert(tk.END, str(0.01))
        self.learning_rate = float(self.Entry_0.get())

        #---- Second label and entry -------
        self.label_1 = tk.Label(win, text='epochs')
        self.label_1.config(font=('Arial', 10))
        self.label_1.place(x=x0, y=y0 + 50)
        self.Entry_1 = tk.Entry()
        self.Entry_1.place(x=xt0, y=y0 + 50)
        self.Entry_1.insert(tk.END, str(1000))
        self.epochs = float(self.Entry_1.get())

        #---- Third label and entry -------
        self.label_2 = tk.Label(win, text='隱藏層模型架構\n(以半形逗號區分每層 neuron 數量)')
        self.label_2.config(font=('Arial', 10))
        self.label_2.place(x=x0, y=y0 + 100)
        self.Entry_2 = tk.Entry()
        self.Entry_2.place(x=xt0-50, y=y0 + 140)
        self.Entry_2.insert(tk.END, str("10, 15"))
        self.model_architecture = str(self.Entry_2.get())

        #---- Compute button -------
        self.btn1 = tk.Button(win, text='開始訓練', command= self.start)
        self.btn1.place(x=xt0, y=y0 + 200)

        self.btn2 = tk.Button(win, text='本次訓練紀錄', command= self.show_history)
        self.btn2.place(x=xt0-10, y=y0 + 230)

        self.btn3 = tk.Button(win, text='成功紀錄', command= self.show_successful_history)
        self.btn3.place(x=xt0, y=y0 + 260)

        self.figure = Figure(figsize=(5, 10), dpi=100)
        self.subplot1 = self.figure.add_subplot(111)

        #---- Show the plot-------
        self.plots = FigureCanvasTkAgg(self.figure, win)
        self.plots.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=0)

        toolbar = NavigationToolbar2Tk(self.plots, win)
        toolbar.update()

        # get date-time ####################################

        loc_dt = datetime.datetime.today()
        datetime_format = loc_dt.strftime("%Y-%m-%d_%H-%M-%S")
        # os.mkdir(my_dir / str('drive_test_' + datetime_format + '/'), 0o666)
        os.mkdir('./drive_test_' + datetime_format + '/', 0o666)
        self.datetime_format = datetime_format
        self.counter = 0

    def start(self):
        self.filename = str(self.Entry_file.get())
        self.learning_rate = float(self.Entry_0.get())
        self.epochs = float(self.Entry_1.get())
        

        loc_dt = datetime.datetime.today()
        datetime_format = loc_dt.strftime("%Y-%m-%d_%H-%M-%S")
        # os.mkdir(my_dir / str('drive_test_' + datetime_format + '/'), 0o666)
        os.mkdir('./drive_test_' + datetime_format + '/', 0o666)
        self.datetime_format = datetime_format

        # get wall ####################################
        #rail_map = np.genfromtxt( my_dir / "wall_coordinate.txt", dtype=int, delimiter=",")
        # file = open( my_dir / "wall_coordinate.txt", "r", encoding="utf-8")
        file = open( "./wall_coordinate.txt", "r", encoding="utf-8")
        lines = file.readlines()
        rail_map = []
        for line in lines:
            rail_map.append( [ int (x) for x in line.split(',') ] )

        print(rail_map)

        phi = rail_map[0][2]
        circle_center = [0, 0]

        # training ####################################

        train_path = []
        # train_path = np.loadtxt( my_dir /  "/train6dAll.txt", dtype=float)
        file = open( './' + self.filename, "r", encoding="utf-8")
        lines = file.readlines()
        train_path = []
        for line in lines:
            train_path.append( [ float (x) for x in line.split(' ') ] )

        
        train_path = np.array(train_path)
        print(np.shape(train_path))
        # print(train_path)

        self.model_architecture = str(self.Entry_2.get())
        model_architecture = [int(x.strip()) for x in self.model_architecture.split(',')]
        model_architecture.insert(0, np.shape(train_path)[1]-1) # 輸入層為 5 (x, y, front, right, left)
        model_architecture.append(1) #輸出層為 1 ,為方向盤角度
        print("模型架構：", model_architecture)
        
        # print("first 5 column")
        # X_train = train_path[:, :5]
        X_train = train_path[:, :np.shape(train_path)[1]-1]
        # print("last 1 column")
        # Y_train = train_path[:, 5:]
        Y_train = train_path[:, np.shape(train_path)[1]-1:]

        # Normalize the input
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(Y_train)
        Y_norm = scaler.transform(Y_train)

        if len(model_architecture) < 4:
            if np.shape(train_path)[1] == 6:
                model_architecture = [5, 10, 15, 1] # apply default
            if np.shape(train_path)[1] == 4:
                model_architecture = [3, 10, 15, 1] # apply default

        neural_net_test = mlp.NeuralNet(model_architecture, batch_size = 1, target='linear-reg')

        neural_net_test.train(X=X_train, Y=Y_norm, r=self.learning_rate,
                    iterations=self.epochs, shuffle=True, check_grad=True)

        y_fitted = neural_net_test.train_fitted()
        # self.subplot1.cla()
        # self.subplot1.plot(neural_net_test.target_history)
        # self.subplot1.set_xlabel("Epoch")
        # self.subplot1.set_ylabel("MSE")
        # self.subplot1.set_title("Trained Result")
        # self.plots.draw()
        # plt.plot(neural_net_test.target_history)
        # plt.xlabel("Epoch")
        # plt.ylabel("MSE")
        # plt.savefig('./drive_test_' + self.datetime_format + '/mse-epoch.png')
        # closePlots(plt)
        #self.subplot1.plot(mlp.NeuralNet.get_history(neural_net_test))
        # self.subplot1.plot(neural_net_test.target_history)

        # initialization ####################################

        timestep_t = 0
        x_prev, y_prev = circle_center[0], circle_center[1]
        phi_prev = phi # 與水平軸的角度 -90 ~ 270
        theta_prev = 0 # 方向盤打的角度 -40 ~ 40
        b = 6 #car_length
        itter = 100
        theta_prediction_with_xy = []

        #####################################

        circle = Circle(xy =(0, 0), radius=3, alpha=0.1, color='r')
        circle_center = [0, 0]

        wall_line, plt, ax =  draw_wall(rail_map, circle_center, phi)
        distance_front_list, distance_left_list, distance_right_list = draw_intersection(wall_line, circle_center, phi)

        #相對水平軸直線
        x1, y1 = rotate(4, 0, math.radians(phi), circle_center[0], circle_center[1])
        plt.plot([circle_center[0], x1], [circle_center[1], y1], color='r')
        # print(x1,", ", y1)
        x_left, y_left = rotate(4, 0, math.radians(phi+45), circle_center[0], circle_center[1])
        plt.plot([circle_center[0], x_left], [circle_center[1], y_left], color='g')
        x_right, y_right = rotate(4, 0, math.radians(phi-45), circle_center[0], circle_center[1])
        plt.plot([circle_center[0], x_right], [circle_center[1], y_right], color='b')

        ax.add_patch(circle)
        self.subplot1.plot = plt
        self.plots.draw()
        # plt.savefig(my_dir / str('drive_test_' + self.datetime_format + '/drivecar_step_0.png'))
        plt.savefig('./drive_test_' + self.datetime_format + '/drivecar_step_0.png')
        closePlots(plt)

        #####################################

        while(arrive_goal(circle_center, rail_map[1], rail_map[2]) == False):
            wall_line, plt, ax =  draw_wall(rail_map, [x_prev, y_prev], phi_prev)

            distance_front_list, distance_left_list, distance_right_list = draw_intersection(wall_line, [x_prev, y_prev], phi_prev)
            if not distance_front_list or not distance_left_list or not distance_right_list:
                print("Collision with wall")
                messagebox.showinfo('預測結果失敗', '此次預測結果失敗，可以重新訓練一次\n或按成功紀錄來看成功的結果')
                break

            if distance_front_list[0] < 3.0 or distance_right_list[0] < 3.0 or distance_left_list[0] < 3.0:
                print("Collision with wall")
                messagebox.showinfo('預測結果失敗', '此次預測結果失敗，可以重新訓練一次\n或按成功紀錄來看成功的結果')
                break

            print("前方距離", distance_front_list)
            print("右方距離", distance_right_list)
            print("左方距離", distance_left_list)

            # 減掉車身長度
            for i in range(len(distance_front_list)):
                distance_front_list[i] = distance_front_list[i] -3
            for i in range(len(distance_right_list)):
                distance_right_list[i] = distance_right_list[i] -3
            for i in range(len(distance_left_list)):
                distance_left_list[i] = distance_left_list[i] -3

            if np.shape(train_path)[1] == 4:
                prediction_theta = neural_net_test.predict(np.array([distance_front_list[0], distance_right_list[0], distance_left_list[0]]))
            if np.shape(train_path)[1] == 6:
                prediction_theta = neural_net_test.predict(np.array([x_prev, y_prev, distance_front_list[0], distance_right_list[0], distance_left_list[0]]))
            
            prediction_theta = prediction_theta[0][0]

            # 超過限制值便拉回來
            if math.degrees(prediction_theta) > 40:
                prediction_theta = math.radians(40)
            elif math.degrees(prediction_theta) < -40:
                prediction_theta = math.radians(-40)

            # prediction_theta = math.radians((math.degrees(prediction_theta) // 10) * 10)
            # prediction_theta = int(input("input theta"))
            theta_prediction_with_xy.append([x_prev, y_prev, distance_front_list[0], distance_right_list[0], distance_left_list[0], math.degrees(prediction_theta)])
            
            if prediction_theta > 0:
                # print("右轉") #正
                ax.text(-7, 50, "Turn right:", fontsize=11, color ="blue", fontweight ='bold') # + str(round(prediction_theta, 5))
            else:
                # print("左轉") #負
                ax.text(-7, 50, "Turn left:", fontsize=11, color ="green", fontweight ='bold') #+ str(round(prediction_theta, 5))

            print("prediction_theta: ", math.degrees(prediction_theta))
            
            x_next = x_prev + math.cos(math.radians(phi_prev) + prediction_theta) + math.sin(prediction_theta) * math.sin(math.radians(phi_prev))
            y_next = y_prev + math.sin(math.radians(phi_prev) + prediction_theta) - math.sin(prediction_theta) * math.cos(math.radians(phi_prev))
            phi_next = phi_prev - math.degrees(math.asin(2*math.sin(prediction_theta)/b))
                    
            ax.text(-7, 47.5, "Phi(t+1):" + str(round(phi_next, 5)), fontsize=11)
            ax.text(-7, 45, "Theta(t+1):" + str(round(prediction_theta, 5)), fontsize=11)

            # print("prediction_theta", math.degrees(prediction_theta))

            circle = Circle(xy =(x_next, y_next), radius=3, alpha=0.1, color='r')
            circle_center = [x_next, y_next]
            #plt.scatter(x_next, y_next, color='lawngreen')
            #相對水平軸直線
            x1, y1 = rotate(4, 0, math.radians(phi_next), circle_center[0], circle_center[1])
            plt.plot([circle_center[0], x1], [circle_center[1], y1], color='r')
            # print(x1,", ", y1)
            x_left, y_left = rotate(4, 0, math.radians(phi_next+45), circle_center[0], circle_center[1])
            plt.plot([circle_center[0], x_left], [circle_center[1], y_left], color='g')
            x_right, y_right = rotate(4, 0, math.radians(phi_next-45), circle_center[0], circle_center[1])
            plt.plot([circle_center[0], x_right], [circle_center[1], y_right], color='b')

            ax.add_patch(circle)

            timestep_t += 1
            x_prev, y_prev = x_next, y_next
            phi_prev = phi_next
            theta_prev
            itter -= 1

            self.subplot1.plot = plt
            self.plots.draw()

            # plt.savefig(my_dir / str('drive_test_' + self.datetime_format + '/drivecar_step_'+ str(timestep_t) +'.png'))
            plt.savefig('./drive_test_' + self.datetime_format + '/drivecar_step_'+ str(timestep_t) +'.png')
            closePlots(plt)

        # 車子行徑記錄
        # file = open(my_dir / str('drive_test_' + self.datetime_format + '/prediction_record_6D.txt'), 'w')
        file = open('./drive_test_' + self.datetime_format + '/prediction_record_6D.txt', 'w')
        for item in theta_prediction_with_xy:
            print(item)
            for string_item in item:
                file.write(str(string_item) + " ")
            file.write("\n")
        file.close()

        file = open('./drive_test_' + self.datetime_format + '/prediction_record_4D.txt', 'w')
        for item in theta_prediction_with_xy:
            print(item)
            for string_item in item[1:]: # Omit x, y coordinate
                file.write(str(string_item) + " ")
            file.write("\n")
        file.close()

        neural_net_test.obj_history = []
        del neural_net_test.obj_history
        del neural_net_test
        messagebox.showinfo('預測成功結束', '訓練結束！可以點擊 "本次訓練紀錄" 查看訓練結果')

    def show_history(self):
        # proc = subprocess.Popen(["python", my_dir / "slide.py", "-d" + str("drive_test_" + self.datetime_format)])
        proc = subprocess.Popen(["python", "./slide.py", "-d" + str("drive_test_" + self.datetime_format)])

    def show_successful_history(self):
        # proc = subprocess.Popen(["python", my_dir / "slide.py"])
        proc = subprocess.Popen(["python", "./slide.py"])

window = tk.Tk()
gui_window = GUI_window(window)
window.title('MLP drive car')
window.geometry("800x400+10+10")
window.mainloop()
window.quit()

########################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
########################################################################