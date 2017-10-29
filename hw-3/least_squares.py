import numpy as np

A=np.matrix([[.16,.1],[.17,.11],[2.02,1.29]])
b=np.matrix([[.26],[.28],[3.31]])

def Gaussian_Elimination(A):
    m=A.shape[0]
    n=A.shape[1]
    if(m!=n):
        print( 'Matrix is not square!');
        return
    for k in range(0,n-1):
        if A[k,k] == 0:
            return
        for i in range(k+1,n):
            A[i,k]=A[i,k]/A[k,k]
        for j in range(k+1,n):
            for i in range(k+1,n):
                A[i,j]-=A[i,k]*A[k,j]

def Back_Substitution(A,b,x):
    m=A.shape[0]
    n=A.shape[1]
    if(m!=n):
        print( 'Matrix is not square!')
        return
    for j in range(n-1,-1,-1):
        if A[j,j] == 0:
            print( 'Matrix is singular!')
            return          # matrix is singular
        x[j] = b[j]/A[j,j]
        for i in range(0,j):
            b[i] = b[i] - A[i,j]*x[j]
			
def Forward_Substitution(A,b,x):
    m=A.shape[0]
    n=A.shape[1]
    if(m!=n):
        print( 'Matrix is not square!')
        return
    for j in range(0,n):
        if A[j,j] == 0:
            print( 'Matrix is singular!')
            return          # matrix is singular
        x[j] = b[j]/A[j,j]
        for i in range(j+1,n):
            b[i] = b[i] - A[i,j]*x[j]
			
AtA=A.transpose().dot(A)
Atb=A.transpose().dot(b)

Gaussian_Elimination(AtA)

L=np.identity(AtA.shape[0])

for i in range(1,L.shape[0]):
	for j in range(0,i):
		L[i,j]=AtA[i,j];

U=np.zeros((AtA.shape[0],AtA.shape[1]))

for i in range(U.shape[0]):
	for j in range(U.shape[1]-1,i-1,-1):
		U[i,j]=AtA[i,j];
		
AtbVector=np.array(Atb[:,0])
y=np.zeros(A.shape[1])
x=np.zeros(A.shape[1])
Forward_Substitution(L,AtbVector,y)
Back_Substitution(U,y,x)
print('x:')
print(x)

#Part B
b=np.matrix([[.27],[.25],[3.33]])
AtA=A.transpose().dot(A)
Atb=A.transpose().dot(b)

Gaussian_Elimination(AtA)

L=np.identity(AtA.shape[0])

for i in range(1,L.shape[0]):
	for j in range(0,i):
		L[i,j]=AtA[i,j];

U=np.zeros((AtA.shape[0],AtA.shape[1]))

for i in range(U.shape[0]):
	for j in range(U.shape[1]-1,i-1,-1):
		U[i,j]=AtA[i,j];
		
AtbVector=np.array(Atb[:,0])
y=np.zeros(A.shape[1])
x=np.zeros(A.shape[1])
Forward_Substitution(L,AtbVector,y)
Back_Substitution(U,y,x)
print('Part B x:')
print(x)

print('K(A^T*A):')
print(np.linalg.cond(A.transpose().dot(A)))