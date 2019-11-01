import tkinter as tk

root = tk.Tk()
button1 = tk.Button(root, text='btn1')
button2 = tk.Button(root, text='btn2')
button3 = tk.Button(root, text='btn3')
button1.grid(row=0, column=0)
button2.grid(row=0, column=1)
button3.grid(row=1, column=1)
