import numpy as np
import math as math

def g1(x):
    g = (pow(x,2) + 2)/3
    return g
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

def fpiters(g, x0, maxiter):
    xp = []
    xp.insert(0,2)
    for i in range(0,maxiter,1):
        t = xp[i]
        x = g(t)
        xp.insert(i+1, x)
        print(xp[i])


def fixedp(f,x0,tol=10e-6,maxiter=100):
    """ Fixed point algorithm """
    e = 1
    itr = 0
    x = 2.5
    xp = []
    while(e < tol and itr < maxiter):
        x = f(x0)      # fixed point equation
        #print(x)
        e = np.linalg.norm(x-x0) # error at the current step
        x0 = x
        xp.append(x0)  # save the for i in range(0, 100)]solution of the current step
        #print(xp[itr])
        itr = itr + 1
    return x,xp

def main():
    g1 = lambda x: (pow(x, 2) + 2)/3
    g2 = lambda x: math.sqrt(3*x - 2)
    g3 = lambda x: 3 - 2/x
    g4 = lambda x: (pow(x,2)-2)/(2*x - 3)

    fpiters(g2, 2, 20)

if __name__ == "__main__":
    main()

