import numpy as np
from numpy import linalg as linalg

def main():
    A = np.matrix([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
    b = np.matrix([[0.1],[0.3],[0.5]])

    x= np.linalg.solve(A,b)
    k_cond = np.linalg.cond(A)
    
    print x
    print 'condition number: ', k_cond
    print(np.finfo(float).eps)

if __name__ == "__main__":
    main()
