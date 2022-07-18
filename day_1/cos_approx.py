#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Aliaa Afify'
__email__ = 'aliaa.afify@gmail.com'

from math import factorial
from math import pi


def cos_approx(x, accuracy=10):
    
    COS = sum( (-1)**n * x**(2*n) / factorial(2*n) for n in range(accuracy+1) )
            
    return COS


# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
