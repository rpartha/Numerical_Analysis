import numpy as np
from numpy import linalg as la
import math as math

def gauss_elim(A):
	m=A.shape[0]
	n=A.shape[1]
	U = np.zeros((m,n))
	L = np.zeros((m,n))

	if(m!=n):
		print('Not Square Matrix')
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
		for i in range(kk+1, n):
			L[kk,kk] = 1
	L[kk,kk]=1

	U = np.triu(A,0)

	return (L,U)

def forward_sub(A,b,x):
	m=A.shape[0]
	n=A.shape[1]
	if(m!=n):
		print 'Matrix is not square!'
		return
	for j in range(0,n):
		if A[j,j] == 0:
			print 'Matrix is singular!'
			return          # matrix is singular
		x[j] = b[j]/A[j,j]
		for i in range(j+1,n):
			b[i] = b[i] - A[i,j]*x[j]

def back_sub(A,b,x):
	m=A.shape[0]
	n=A.shape[1]
	if(m!=n):
		print 'Matrix is not square!'
		return
	for j in range(n-1,-1,-1):
		if A[j,j] == 0:
			print 'Matrix is singular!'
			return          # matrix is singular
		x[j] = b[j]/A[j,j]
		for i in range(0,j):
			b[i] = b[i] - A[i,j]*x[j]

def main():
	x_exact = np.array([1,1])
	for k in range(1,11):
		eps = math.pow(10, -2*k)
		print 'epsilon: ', eps
		A = np.matrix([[eps,1],[1,1]])
		b = np.array([(1+eps),2])
		L,U = gauss_elim(A)
		x=np.zeros(2)
		y=np.zeros(2)
		forward_sub(L,b,y)
		back_sub(U,y,x)
		print 'error: ', la.norm([(x[i]-x_exact[i]) for i in range (0,2)])

if __name__ == "__main__":
	main()