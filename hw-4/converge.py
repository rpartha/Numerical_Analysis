import numpy as np
import math as math

"""
def fpiters(g,x_true, eps):
    numIters = 10
    step = 0
    x = 2.5
    err = np.fabs(x - x_true)
    #sol = []

    while((err > eps) and (step < numIters)):
        step += 1
        x = g(x_true)
        err_new = np.linalg.norm(x - x_true)
        ratio = err_new/err
        err = err_new
        print(x)
        #print(step, err, ratio)
    return x
"""

def fpiters(g, x0, n):
    xp = []
    xp.insert(0,2)
    for i in range(0,n,1):
        print('iteration: ', i)
        t = xp[i]
        x = g(t)
        xp.insert(i+1, x)
        print(xp[i])
    return x, xp


def fixedp(f,x0,eps=10e-6,n=100):
    """ Fixed point algorithm """
    e = 1
    itr = 0
    xp = []
    x = 2.5
    while(e > eps):
        x = f(x0)     
        #print(x)
        e = np.linalg.norm(np.fabs(x-x0))
        x0 = x
        xp.append(x0) 
        #print(xp[itr])
        itr = itr + 1
        print(itr)
    return x,xp

def main():
    g1 = lambda x: (pow(x, 2) + 2)/3
    g2 = lambda x: math.sqrt(3*x - 2)
    g3 = lambda x: 3 - 2/x
    g4 = lambda x: (pow(x,2)-2)/(2*x - 3)

    x,xp = fpiters(g1, 2.1, 100)

    print(x)

if __name__ == "__main__":
    main()

