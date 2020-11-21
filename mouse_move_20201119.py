# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:22:54 2020

@author: takeshi.goto

https://qiita.com/konitech913/items/301bb63c8a69c3fcb1bd

https://chindafalldesu.hatenablog.com/entry/2020/01/05/001349

https://qiita.com/deaikei/items/7f1acaa3b1db40c33f1a

"""

import ctypes
import pyautogui
import time
import datetime

mouse_posi=pyautogui.position()
print("mouse_posi",mouse_posi)

utime=datetime.datetime.now().timestamp()

while True:
    #time.sleep(5)
    mouse_posi02=pyautogui.position()
    with open("utime_file.txt","r") as f:
        txt01=f.read()
        utime02=float(txt01.rstrip("\r\n"))

    if utime02 != utime :
        #
        print("if_a",datetime.datetime.now())
        pass
        
    elif mouse_posi != mouse_posi02:
        #
        print("if_b",datetime.datetime.now())
        pass

    else:
        pyautogui.moveTo(20000,300, duration=0) 
        pyautogui.click()
        pyautogui.moveTo(1500,300, duration=0) 
        #
        print("if_c",datetime.datetime.now())
    
    time.sleep(100)
    #time.sleep(3)
    mouse_posi=mouse_posi02
    utime=utime02
            
            





