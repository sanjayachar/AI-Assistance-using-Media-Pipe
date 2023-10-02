from tkinter import *
from PIL import ImageTk,Image
import numpy as np
import matplotlib.pyplot as plt
import PoseEstimationCurl as curl
import countDown as cnt
root=Tk()
root.title('Sanjay.com')
root.iconbitmap('')
root.geometry("400x200")

def graph():
    Angle=cnt.count
    count=curl.counts
    plt.plot(Angle,count)
    plt.show()
my_Btn=Button(root,text="Graph It!",command=graph)
my_Btn.pack()
root.mainloop()