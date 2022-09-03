#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
９０度ずつ回転し、４パターンの図のどれにでも存在するものが答えとなる。

"""




a,b=[int(x) for x in input().split()]

ls00=list()

for i in range(a):
    ls00.append([ 1 if x=='B' else 0 for x in list(input())])

#########################


# a=5
# b=7


# # ls00=[[0, 0, 0, 0, 0, 0, 0], 
# #       [0, 0, 0, 0, 0, 0, 0], 
# #       [0, 0, 0, 0, 0, 0, 0], 
# #       [0, 0, 0, 0, 0, 0, 0], 
# #       [0, 0, 0, 0, 0, 0, 0]]


# ls00=[[1, 1, 1, 1, 0, 0, 1], 
#       [1, 1, 1, 1, 0, 1, 0], 
#       [0, 0, 0, 0, 1, 0, 0], 
#       [0, 1, 1, 1, 0, 1, 1], 
#       [0, 0, 1, 1, 0, 1, 1]]


ls99=[ x[::-1] for x in ls00][::-1]


import numpy as np

ls100=np.array(ls00).T.tolist()

ls199=[ x[::-1] for x in ls100][::-1]





def f01(ls_, a, b, cond):
    ls_res=[]
    
    for i in range(a):
        for j in range(b):
            if ls_[i][j]==1 and (i==0 or ls_[i-1][j]==0) and (j==0 or ls_[i][j-1]==0):
                ls01=list()
                for ii in range(i,a):
                    if ls_[ii][j]==1:
                        for jj in range(j,b):
                            if ls_[ii][jj]==0 :
                                break
                            else:
                                if cond ==0:
                                    ls01.append([ii,jj])
                                elif cond==1:
                                    ls01.append([a-ii-1,b-jj-1])
                                elif cond==2:
                                    ls01.append([jj,ii])
                                elif cond==3:
                                    ls01.append([b-jj-1,a-ii-1])
    
                    else:
                        break
    
            try:
                if ls01 not in ls_res:
                    ls_res.append(ls01)
            except:
                pass  
    return ls_res
    

ls_res00=f01(ls00,a,b,0)
ls_res99=f01(ls99,a,b,1)

ls_res100=f01(ls100,b,a,2)
ls_res199=f01(ls199,b,a,3)


ls_res99=[ x[::-1] for x in ls_res99][::-1]
ls_res199=[ x[::-1] for x in ls_res199][::-1]


dt_res00={  '_'.join(sorted([ '{0}-{1}'.format(y[0],y[1]) for y in x ])) for x in ls_res00}
dt_res99={  '_'.join(sorted([ '{0}-{1}'.format(y[0],y[1]) for y in x ])) for x in ls_res99}
dt_res100={  '_'.join(sorted([ '{0}-{1}'.format(y[0],y[1]) for y in x ])) for x in ls_res100}
dt_res199={  '_'.join(sorted([ '{0}-{1}'.format(y[0],y[1]) for y in x ])) for x in ls_res199}



print(len( dt_res00 & dt_res99 & dt_res100 & dt_res199 ))

