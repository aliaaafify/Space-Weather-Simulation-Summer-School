#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:54:09 2022

@author: aliaaafify
"""
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description ='My assignment2')
    parser.add_argument(
        '-input',
        help='into iutput file',
        type=str)
    parser.add_argument(
        '-output',
        help='into output file',
        type=str,default='output.png')
        
    args = parser.parse_args()
    return args

def read_ascii_file(filename = '/Users/aliaaafify/Documents/SWSSS/omni_min_def_pTiK8HlHDi.lst', index=-1):
    "this is reading an ascii file of omni data"
    with open(filename) as f:
        data_dic = {"time":[],
                    "year":[],
                    "day":[],
                    "hour":[],
                    "minute":[],
                    "symh":[]}        
        
        for line in f:
            tmp = line.split()

# create datetime in each line
            time0 = dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0) + dt.timedelta(days = int(tmp[1])-1)
            print(time0)
            data_dic["time"].append(time0)
            data_dic["year"].append(int(tmp[0]))
            data_dic["day"].append(int(tmp[1]))
            data_dic["hour"].append(int(tmp[2]))
            data_dic["minute"].append(int(tmp[3]))
            data_dic["symh"].append(int(tmp[4]))
        
    return data_dic

args = parse_args()
print(args)


#filename='/Users/aliaaafify/Documents/SWSSS/omni_min_def_pTiK8HlHDi.lst'
filename = args.input
index = -1

data_dic = read_ascii_file(filename,index)
print(data_dic["time"])

T=data_dic["time"]
D=data_dic["symh"]

T=np.array(T)
D=np.array(D)

fig,ax = plt.subplots()

ax.plot(T,D,marker='.' , c='gray', label='All events, alpha=0.5')
lp = D <-100
print(lp)


ax.plot(T[lp],D[lp], marker='+',
        linestyle='',
        c= 'tab:orange',
        label='<-100 nT',
        alpha=0.6)

#ax.vlines([20, 100], 0, 1, linestyles='dashed', colors='red')

ax.set_xlabel('year of 2013')
ax.set_ylabel('SYMH (nT)')
ax.grid(True)
ax.legend()

outfile = args.output
print('Writing file : ' + outfile)
plt.savefig(outfile)
plt.close()

minsymh=np.argmin(D)
print("minsymh ="+str(D[minsymh]))
print("time of minsymh ="+str(T[minsymh].isoformat()))