# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 20:12:35 2020

@author: takeshi.goto
"""

import ctypes
import datetime
import time

while True:
    for i in range(0, 0xff):
        if ctypes.windll.user32.GetAsyncKeyState(i)==-32768 and i in [0x01,0x02,0x04,0x08,0x09,0x0D,0x10,0x11,0x12] :
            #print("sss")
            aa=datetime.datetime.now()
            with open("utime_file.txt","w") as f:
                print(aa.timestamp(),file=f)
        elif ctypes.windll.user32.GetAsyncKeyState(i)==32768 and ( (i in [0x0D, 0x08]) or (i>=0x30 and i<=0x39) or (i>=0x41 and i<=0x5A)) :
            #print(chr(i))
            aa=datetime.datetime.now()
            with open("utime_file.txt","w") as f:
                print(aa.timestamp(),file=f)   
        time.sleep(0.001)




