import numpy as np

#ClassNotes GaussianElimination
def gaussianElimination(A):
    m=A.shape[0]
    n=A.shape[1]
    U = np.zeros((m,n))
    L = np.zeros((m,n))

    if(m!=n):
        print 'Not Square Matrix'
        return

    for k in range(0, n-1):
        if A[k,k] == 0:
            return
        for i in range(k+1, n):
            A[i,k] = A[i,k] / A[k,k]
        for j in range (k+1, n):
            for i in range (k+1, n):
                A[i,j] -= A[i,k] * A[k,j]

    L = np.tril(A,0)
    for kk in range(n):
        for i in range(kk + 1, n):
            L[kk, kk] = 1
    L[kk, kk] = 1

    U = np.triu(A,0)

#HomeWork EliminationMatrices
def eliminationMatrices(A,E):
    m = A.shape[0]
    n = A.shape[1]
    U = np.copy(A)
    L = np.zeros((m, n))

    if(m!=n):
        print 'Not Square Matrix'
        return

    for k in range(n):
        
        maxi = abs(A[k:, k]).argmax() + k
        if maxi != k:
            U[[k, maxi]] = U[[maxi, k]]
            L[[k, maxi]] = L[[maxi, k]]
            E[[k, maxi]] = E[[maxi, k]]
            
        L[k,k] = 1
            
        for j in range(k+1, m):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:] -= L[j, k] * U[k, k:]
            E[j] -= L[j, k] * E[k]
            

    X = np.zeros(m)
    k = n-1
    X[k] = E[k]/A[k, k]
    while k >= 0:
        X[k] = (E[k] - np.dot(U[k, k+1:], X[k+1:]))/U[k, k]
        k -= 1

    print '-------L-------'
    print np.matrix.round(L, 3)
    print '-------U-------'
    print np.matrix.round(U, 3)
    print '-----Vector----'
    print X

def main():
    # ClassExample
    # A=np.matrix([[2.0,4.0,-2.0], [4.0,9.0,-3.0], [-2.0,-3.0,7.0]])
    # E=np.matrix([[2.0],[8.0],[10.0]])
    A=np.matrix([[21.0,32.0,14.0,8.0,6.0,9.0,11.0,3.0,5.0],
                 [17.0,2.0,8.0,14.0,55.0,23.0,19.0,1.0,6.0],
                 [41.0,23.0,13.0,5.0,11.0,22.0,26.0,7.0,9.0],
                 [12.0,11.0,5.0,8.0,3.0,15.0,7.0,25.0,19.0],
                 [14.0,7.0,3.0,5.0,11.0,23.0,8.0,7.0,9.0],
                 [2.0,8.0,5.0,7.0,1.0,13.0,23.0,11.0,17.0],
                 [11.0,7.0,9.0,5.0,3.0,8.0,26.0,13.0,17.0],
                 [23.0,1.0,5.0,19.0,11.0,7.0,9.0,4.0,16.0],
                 [31.0,5.0,12.0,7.0,13.0,17.0,24.0,3.0,11.0]])
    E=np.matrix([[2.0],[5.0],[7.0],[1.0],[6.0],[9.0],[4.0],[8.0],[3.0]])
    eliminationMatrices(A,E)

if __name__ == '__main__':
        main()