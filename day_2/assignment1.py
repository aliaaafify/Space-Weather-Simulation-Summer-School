#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Aliaa Afify'
__email__ = 'aliaa.afify@gmail.com'

from math import factorial
from math import pi
import argparse
import numpy as np


def cos_approx(x, accuracy=10):
   """ Python code to demonstrate the working of cos()"""
    COS = sum( (-1)**n * x**(2*n) / factorial(2*n) for n in range(accuracy+1) )
    return COS
 
def parse_args():
    parser = argparse.ArgumentParser(description ='My assignment')
    parser.add_argument('x',type = float, help = 'insert the value of x (type 2 value)', default=0.0)
    parser.add_argument('npts', help = 'insert the number of points ', type = int, default = 10)
    args = parser.parse_args()
    return args

args = parse_args()
print(args)
x = args.x
print(x)
npts = args.npts
print(npts)
result = cos_approx(x,npts)
print(result)

python_result=np.cos(x)
print(python_result)

judgment = python_result-0.0001 <= result and python_result+0.0001 >= result
print(judgment)

"""
0.0  1.0
1.57  0.0007963267107333618
3.14  -0.9999987316527259
-1.57  0.0007963267107332633
-3.14  -0.9999987316527259
6.28  1.0002928297593128
"""