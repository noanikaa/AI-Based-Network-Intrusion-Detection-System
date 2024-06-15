from tkinter import *
from tkinter import ttk
import time
import xlsxwriter
from datetime import date
import sys
import os
from os import listdir
from os.path import isfile, join

import xlrd
from tkinter import filedialog
from tkinter import messagebox
import tkinter.messagebox

import math
from collections import Counter
import pandas as pd
import numpy as np
import numpy
import random


seed = 7
numpy.random.seed(seed)


import argparse

window = Tk()
window.title("IDS FOR NETWORK SECURITY")
window.geometry('700x300')

tab_control = ttk.Notebook(window)
tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='LOGIN')

#############################################################################################################################################################

def TST_SPEECH():
   Un=ee1.get()
   Pw=ee2.get()
   print('USERNAME',Un)
   wb = xlrd.open_workbook('demo.xlsx') 
   sheet = wb.sheet_by_index(0) 
   Un1=sheet.cell_value(1, 0)
   Pw1=sheet.cell_value(1, 1)
   print('UN',Un1);
   print('PW',Pw1);
   if Un==Un1 and Pw==Pw1:
      messagebox.showinfo('LOGIN SUCCESSFUL', 'WELCOME')
      TRR=[1,1,1,1,1];
      np.savetxt('aa.txt',TRR)
      lbl2.configure(text='SUCCESSFUL LOGIN')
      import GUI_IDS_Home
   else:
      TRR=[2,1,1,1,1];
      np.savetxt('aa.txt',TRR)
      lbl2.configure(text=' ACCESS DENIED')
      messagebox.showerror('LOGIN DENIED', 'Wrong Username Or Password')
      window.quit()
      window.destroy()

def RST():
      messagebox.showerror('CLOSE', 'CLOSE')
      window.quit()
      window.destroy()



# INPUT FORM:diaryofagameaddict.com
   
def GEN_SPEECH1():
    KK=np.loadtxt('aa.txt')
    if KK[0]==1:
       lbl2.configure(text='TRAINING--')
       vectorizer, model  = TL(1)
    else:
       lbl1.configure(text='ACCESS DENIED')
          


 	 
#############################################################################################################################################################
lbl = Label(tab2, text="LOGIN",font=("Arial Bold", 30),foreground =("blue"))
lbl.grid(column=0, row=0)
lbl = Label(tab2, text="FORM",font=("Arial Bold", 30),foreground =("blue"))
lbl.grid(column=1, row=0)
# USERNAME & PASSWORD ENTRY BOX
Label(tab2, text="USERNAME",font=("Arial Bold", 15),foreground =("red")).grid(row=1,column=0,sticky=W, padx=20, pady=20)
Label(tab2, text="PASSWORD",font=("Arial Bold", 15),foreground =("red")).grid(row=2,column=0,sticky=W,  padx=20, pady=20)

ee1 = Entry(tab2)
ee2 = Entry(tab2,show='*')

ee1.grid(row=1, column=1,sticky=W, padx=20, pady=20)
ee2.grid(row=2, column=1,sticky=W, padx=20, pady=20)


lbl2= Label(tab2, text="  STATUS   ",font=("Arial Bold", 10),foreground =("black"),background  =("white"))
lbl2.grid(column=4, row=1)

Button(tab2, text='LOGIN', command=TST_SPEECH,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=3, column=1, sticky=W, pady=4)

Button(tab2, text='CANCEL', command=RST,font=("Arial Bold", 15),foreground =("green"),background  =("yellow")).grid(row=3, column=0, sticky=W, pady=4)

#############################################################################################################################################################
tab_control.pack(expand=1, fill='both')
window.mainloop()
