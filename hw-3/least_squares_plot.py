import numpy as np
import matplotlib.pyplot as plt

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

def main():
    t=np.array([0.,1.,2.,3.,4.,5.]);
    y=np.array([1.,2.7,5.8,6.6,7.5,9.9]);
    plt.figure(1);
    plt.plot(t,y,'ro')
    plt.title('Given data');
    plt.draw()

    for n in range(6):
    	
    	A=np.zeros((6,n+1));
    	
    	for r in range(6):
    		for c in range(A.shape[1]):
    			A[r,c]=t[r]**(A.shape[1]-1-c);
    	
    	x=np.zeros(A.shape[1]);
    	y1=np.zeros(A.shape[0]);
    	AtA=A.transpose().dot(A)
    	sol=A.transpose().dot(y)
    	
    	Gaussian_Elimination(AtA)
    	L=np.identity(AtA.shape[0])

    	for i in range(1,L.shape[0]):
    		for j in range(0,i):
    			L[i,j]=AtA[i,j];

    	U=np.zeros((AtA.shape[0],AtA.shape[1]))

    	for i in range(U.shape[0]):
    		for j in range(U.shape[1]-1,i-1,-1):
    			U[i,j]=AtA[i,j];
    	
    	Forward_Substitution(L,sol,y1)
    	Back_Substitution(U,y1,x)
    	x=np.array(x)
    	
    	print('n:',n)
    	print(x)

    	plt.figure(n+2);
    	plt.plot(t,A.dot(x),'k',t,y,'ro')
    	plt.title('n='+str(n));
    	plt.draw()
    plt.show()

if __name__ == "__main__":
    main()

