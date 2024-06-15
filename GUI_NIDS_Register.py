from tkinter import *
from tkinter import ttk
import time
import xlsxwriter
from datetime import date
import xlrd
from tkinter import filedialog
from tkinter import filedialog
from tkinter import messagebox
import tkinter.messagebox
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import sys
import math
from collections import Counter
import pandas as pd
import numpy as np
import numpy
import random

seed = 7
numpy.random.seed(seed)


import argparse

workbook = xlsxwriter.Workbook('demo.xlsx')
worksheet = workbook.add_worksheet()

#worksheet.set_column('A:A', 20)
bold = workbook.add_format({'bold': True})
worksheet.write('A1', 'USERNAME')
worksheet.write('B1', 'PASSWORD')
worksheet.write('C1', 'MOBILE NUMBER')
worksheet.write('D1', 'ROLL NUMBER')
worksheet.write('E1', 'EMAIL ID')

window = Tk()
window.title("IDS for NETWORK SECURITY")
window.geometry('1200x500')

tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab_control.add(tab1, text='USER REGISTRATION')

#############################################################################################################################################################
# HEADING
def show_entry_fields():
   print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))
   Un=e1.get()
   Pw=e2.get()
   print((Un))
   res = "PERSON " + Un + " IS ADDED"
   lbl1.configure(text= res)
   worksheet.write(str('A'+ str(2)),str(Un) )
   worksheet.write(str('B'+ str(2)),str(Pw) )
   workbook.close()

def RST():
      messagebox.showerror('CLOSE', 'CLOSE')
      window.quit()
      window.destroy()

 	 
#######################################################################################################
lbl = Label(tab1, text="                     ",font=("Arial Bold", 30),foreground =("red"))
lbl.grid(column=0, row=0)
lbl = Label(tab1, text="STUDENT  REGISTRATION  DETAILS",font=("Arial Bold", 30),foreground =("red"),background  =("white"))
lbl.grid(column=1, row=0)
lbl = Label(tab1, text="                  ",font=("Arial Bold", 30),foreground =("red"))
lbl.grid(column=2, row=0)
# USERNAME & PASSWORD ENTRY BOX
Label(tab1, text="USERNAME",font=("Arial Bold", 15),foreground =("green")).grid(row=1,column=0)
Label(tab1, text="PASSWORD",font=("Arial Bold", 15),foreground =("green")).grid(row=2,column=0)
Label(tab1, text="MOBILE NUMBER",font=("Arial Bold", 15),foreground =("green")).grid(row=3,column=0)
Label(tab1, text="ROLL NUMBER",font=("Arial Bold", 15),foreground =("green")).grid(row=4,column=0)
Label(tab1, text="EMAIL ID",font=("Arial Bold", 15),foreground =("green")).grid(row=5,column=0)
e1 = Entry(tab1)
e2 = Entry(tab1,show='*')
e3 = Entry(tab1)
e4 = Entry(tab1)
e5 = Entry(tab1)
e1.grid(row=1, column=1,sticky=W, pady=20)
e2.grid(row=2, column=1,sticky=W, pady=20)
e3.grid(row=3, column=1,sticky=W, pady=20)
e4.grid(row=4, column=1,sticky=W, pady=20)
e5.grid(row=5, column=1,sticky=W, pady=20)

lbl1 = Label(tab1, text="  STATUS   ",font=("Arial Bold", 15),foreground =("red"),background  =("white"))
lbl1.grid(column=2, row=4,sticky=W, pady=20)
Button(tab1, text='CANCEL', command=RST,font=("Arial Bold", 15),foreground =("yellow"),background  =("brown")).grid(row=3, column=2, sticky=W, pady=20)
Button(tab1, text='REGISTER', command=show_entry_fields,font=("Arial Bold", 15),foreground =("yellow"),background  =("brown")).grid(row=2, column=2, sticky=W, pady=4)

#############################################################################################################################################################
tab_control.pack(expand=1, fill='both')
window.mainloop()
