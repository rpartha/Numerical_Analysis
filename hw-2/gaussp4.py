import numpy as np
from numpy import linalg as la
import math as math

#problem 4

#in-place gaussian elimination
def gauss_elim_32(A):
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
			A[i,k] = np.float32(A[i,k]) / np.float32(A[k,k])
		for j in range (k+1, n):
			for i in range (k+1, n):
				A[i,j] = np.float32(A[i,j]) -  \
				         np.float32(np.float32(A[i,k]) * np.float32(A[k,j]))

	L = np.tril(A,np.float32(0))
	for kk in range(n):
		for i in range(kk+1, n):
			L[kk,kk] = np.float32(1)
	L[kk,kk]=np.float32(1)

	U = np.triu(A,np.float32(0))

	return (L,U)

#forward substitution to solve Lx = b
def forward_sub_32(A,b,x):
	m=A.shape[0]
	n=A.shape[1]
	if(m!=n):
		print 'Matrix is not square!'
		return
	for j in range(0,n):
		if A[j,j] == 0:
			print 'Matrix is singular!'
			return          # matrix is singular
		x[j] = np.float32(b[j])/np.float32(A[j,j])
		for i in range(j+1,n):
			b[i] = np.float32(b[i]) - np.float32(np.float32(A[i,j])*np.float32(x[j]))

#backwards substitution to solve Ux = b
def back_sub_32(A,b,x):
	m=A.shape[0]
	n=A.shape[1]
	if(m!=n):
		print 'Matrix is not square!'
		return
	for j in range(n-1,-1,-1):
		if A[j,j] == 0:
			print 'Matrix is singular!'
			return          # matrix is singular
		x[j] = np.float32(b[j])/np.float32(A[j,j])
		for i in range(0,j):
			b[i] = np.float32(b[i]) - np.float32(np.float32(A[i,j])*np.float32(x[j]))

#compute the residual vector
def resid(A, b, x):
	d = np.dot(A,x) 
	r = np.asmatrix(b) - np.dot(A, x)
	return r.astype(np.float32)

def main():
	
	A = np.matrix([[21.0,67.0,88.0,73.0],[76.0,83.0,7.0,20.0],
				  [0.0,85.0,56.0,54.0],[19.3,43.0,30.2,29.4]])
	A_copy = A.copy()
	b = np.array([141.0,109.0,218.0,93.7])

	A_single = A.astype(np.float32)
	b_single = b.astype(np.float32)

	L,U = gauss_elim_32(A_single)

	L_single = L.astype(np.float32)
	U_single = U.astype(np.float32)

	epsilon = 10e-15

	x = np.zeros(4)
	diff = np.zeros(4)
	y = np.zeros(4)

	#compute Ly = b
	forward_sub_32(L_single,b_single,y)
	y.astype(np.float32)

	#compute Ux = y
	back_sub_32(U_single,y,x)
	x.astype(np.float32)

	x_old = x 
	x_new = x_old
	norm = 10

	while(norm > epsilon):

		#compute residual
		r = resid(A_copy,b,x_new)
		print 'resid: ', r

		y = np.zeros(4)
		z = np.zeros(4)

		#compute Ly = r
		forward_sub_32(L_single,np.squeeze(np.asarray(r)),y)
		y.astype(np.float32)

		#compute Uz = y
		back_sub_32(U_single,y,z)
		z.astype(np.float32)

		print 'x_old: ', [x_old[i] for i in range(0,4)]
		
		#compute x_new
		x_new = [(x_old[i] + z[i]) for i in range(0,4)]

		print 'x_new: ', [x_new[i] for i in range(0,4)]

		#compute difference of new and old vectors
		diff = [(x_new[i] - x_old[i]) for i in range(0,4)]

		#compute the L_infinity norm
		norm = la.norm(diff, np.inf)

		x_old = x_new 

		print 'norm: ', norm

if __name__ == "__main__":
	main()
