import numpy as np
import math as math
from scipy.linalg import lu

#return max of sum of columns
def computeL1norm(A):
    column_sums = [sum([math.fabs(row[i]) for row in A]) for i in range(0,len(A[0]))]
    max = column_sums[0]
    for i in range(len(column_sums)):
        if(i > max):
            max = i
    return max

def solve_lower(U,v):
    m=U.shape[0]
    n=U.shape[1]
    c = [0 for i in range(0,n)]

    if(m!=n):
        print('Matrix is not square')
        return
    for j in range(0,n):
        if(U[j,j] == 0):
            print('Matrix is singular')
            return
        v[j] = max(np.absolute(c[j]-1/U[j,j]),np.absolute(c[j]+1/U[j,j])) 
        #v[j] = c[j]/U[j,j]
        for i in range(j+1,n):
            c[i] = c[i] - U[i,j]*v[j]
    
def main():
    A1 = np.array([[10.,-7.,0.],[5.,-1.,5.],[-3.,2.,6.]])
    normA1 = computeL1norm(A1)
    
    
    P,L,U = lu(A1,permute_l=False)
    U_t = U.transpose()
    L_t = L.transpose()
    A_t = U_t.dot(L_t)

    m = U_t.shape[1]
    #print(U_t)

    v = np.zeros(m)
    y = np.zeros(m)
    z = np.zeros(m)

    solve_lower(U_t, v)
    y = np.linalg.solve(L_t, v)
    z = np.linalg.solve(A1, y)

    k = (np.linalg.norm(z,1)/np.linalg.norm(y,1)) * normA1
    
    print('real condition number for A1: ', np.linalg.cond(A1,1))
    print('computed condition number for A1: ', k)

    A2 = np.array([[92.,66.,25.],[-73.,78.,24.],[-80.,37.,10.]])
    normA2 = computeL1norm(A2)

    P,L,U = lu(A2,permute_l=False)
    U_t = np.transpose(U)
    L_t = np.transpose(L)
    m = U_t.shape[1]
    #print(U_t)

    v = np.zeros(m)
    y = np.zeros(m)
    z = np.zeros(m)

    solve_lower(U_t, v)

    y = np.linalg.solve(L_t, v)
    z = np.linalg.solve(A2, y)

    k = (np.linalg.norm(z,1)/np.linalg.norm(y,1)) * normA2
    
    print('real condition number for A2: ', np.linalg.cond(A2,1))
    print('computed condition number for A2: ', k)

if __name__ == "__main__":
    main()