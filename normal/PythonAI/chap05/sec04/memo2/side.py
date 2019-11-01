import tkinter as tk

root = tk.Tk()
button1 = tk.Button(root, text='btn1', width=20).pack()
button2 = tk.Button(root, text='btn2').pack(side=tk.LEFT)
button3 = tk.Button(root, text='btn3').pack(side=tk.RIGHT)
button4 = tk.Button(root, text='btn4').pack(side=tk.BOTTOM)
