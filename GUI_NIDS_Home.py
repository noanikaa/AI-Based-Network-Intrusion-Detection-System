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
window.title("NETWORK SECURITY")
window.geometry('500x500')

tab_control = ttk.Notebook(window)
tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='LOGIN')

#############################################################################################################################################################


def RST():
      messagebox.showerror('CLOSE', 'CLOSE')
      window.quit()
      window.destroy()


# INPUT FORM:diaryofagameaddict.com
   
def RUNMODEL():
    KK=np.loadtxt('aa.txt')
    if KK[0]==1:
       lbl2.configure(text='TRAINING--')
       #vectorizer, model  = TL(1)
       import GIVEN_RF_6_2_20
    else:
       lbl1.configure(text='ACCESS DENIED')
          
def RUNMODEL():
    KK=np.loadtxt('aa.txt')
    if KK[0]==1:
       lbl2.configure(text='TRAINING--')
       #vectorizer, model  = TL(1)
       import GIVEN_RF_TRN
    else:
       lbl1.configure(text='ACCESS DENIED')
       
def RUNMODEL1():
    KK=np.loadtxt('aa.txt')
    if KK[0]==1:
       lbl2.configure(text='TRAINING--')
       #vectorizer, model  = TL(1)
       import ACC_RF
    else:
       lbl1.configure(text='ACCESS DENIED')

def AGIVEN_PLT1():
      import GIVEN_PLT1
       
def BGIVEN_PLT2():
      import GIVEN_PLT2

def Comp_code():
      import Comp_code_2
      
#############################################################################################################################################################
lbl = Label(tab2, text="INTRUSION",font=("Arial Bold", 30),foreground =("red"))
lbl.grid(column=0, row=0)
lbl = Label(tab2, text="DETECTION",font=("Arial Bold", 30),foreground =("red"))
lbl.grid(column=1, row=0)
lbl = Label(tab2, text="",font=("Arial Bold", 30),foreground =("red"))
lbl.grid(column=2, row=0)


lbl2= Label(tab2, text="  STATUS   ",font=("Arial Bold", 10),foreground =("black"),background  =("white"))
lbl2.grid(column=1, row=2)

Button(tab2, text='TEST', command=RUNMODEL,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=2, column=0, sticky=W, pady=4)
Button(tab2, text='ACCURACY', command=RUNMODEL1,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=3, column=0, sticky=W, pady=4)
##Button(tab2, text='TEST CONDITION:3', command=RUNMODEL,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=2, column=1, sticky=W, pady=4)
##Button(tab2, text='TEST CONDITION:4', command=RUNMODEL,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=2, column=2, sticky=W, pady=4)
##Button(tab2, text='TEST CONDITION:5', command=RUNMODEL,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=3, column=1, sticky=W, pady=4)
##Button(tab2, text='TEST CONDITION:6', command=RUNMODEL,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=3, column=2, sticky=W, pady=4)

Button(tab2, text='CANCEL', command=RST,font=("Arial Bold", 15),foreground =("red"),background  =("green")).grid(row=7, column=1, sticky=W, pady=4)
Button(tab2, text='COMPARE', command=Comp_code,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=6, column=0, sticky=W, pady=4)
Button(tab2, text='FEATURE PLOT', command=AGIVEN_PLT1,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=4, column=0, sticky=W, pady=4)
Button(tab2, text='CLASS PROBABILITY', command=BGIVEN_PLT2,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=5, column=0, sticky=W, pady=4)

Button(tab2, text='TRAIN', command=RUNMODEL1,font=("Arial Bold", 15),foreground =("red"),background  =("yellow")).grid(row=1, column=0, sticky=W, pady=4)

#############################################################################################################################################################
tab_control.pack(expand=1, fill='both')
window.mainloop()
