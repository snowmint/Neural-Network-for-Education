from itertools import cycle
import tkinter as tk
#####################################
import os, glob
import re
from pathlib import Path
from PIL import Image, ImageTk
import argparse

my_dir = Path(__file__).parent

def get_parser():
    parser = argparse.ArgumentParser(description='input the pic directory')
    parser.add_argument('-d', '--directory', default='successful_result/[成功]drive_test_2021-12-07_11-58-27')
    return parser

class App(tk.Tk):
    '''Tk window/label adjusts to size of image'''
    def __init__(self, image_files, x, y, delay):
        tk.Tk.__init__(self)
        self.geometry('+{}+{}'.format(x, y))
        self.delay = delay
        self.pictures = cycle((tk.PhotoImage(file=image), image)
                              for image in image_files)
        self.picture_display = tk.Label(self)
        self.picture_display.pack()

    def show_slides(self):

        img_object, img_name = next(self.pictures)
        self.picture_display.config(image=img_object)
        self.title(img_name)
        self.after(self.delay, self.show_slides)

    def run(self):
        self.mainloop()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('directory: ' + args.directory)

    delay = 100

    # os.chdir(my_dir / str(args.directory))
    os.chdir("./" + str(args.directory))
    image_files = []
    for file in glob.glob("drivecar*.png"):
        image_files.append(file)

    image_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    print(image_files)
    photos = cycle(ImageTk.PhotoImage(Image.open(image)) for image in image_files)

    x = 100
    y = 50

    app = App(image_files, x, y, delay)
    app.show_slides()
    app.run()