# Homework #2, 10/17/17
Ramaseshan Parthasarathy  
Akhil Velagapudi  
Tarun Sreenathan  

## Problem 1

The proof is as shown below:

<img src = "../homework-2/p1_proof.png">

## Problem 2

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;21&space;&&space;32&space;&&space;14&space;&&space;8&space;&&space;6&space;&&space;9&space;&&space;11&space;&&space;3&space;&&space;5\\&space;17&space;&&space;2&space;&&space;8&space;&&space;14&space;&&space;55&space;&&space;23&space;&&space;19&space;&&space;1&space;&&space;6\\&space;41&space;&&space;23&space;&&space;13&space;&&space;5&space;&&space;11&space;&&space;22&space;&&space;26&space;&&space;7&space;&&space;9\\&space;12&space;&&space;11&space;&&space;5&space;&&space;8&space;&&space;3&space;&&space;15&space;&&space;7&space;&&space;25&space;&&space;19\\&space;14&space;&&space;7&space;&&space;3&space;&&space;5&space;&&space;11&space;&&space;23&space;&&space;8&space;&&space;7&space;&&space;9\\&space;2&space;&&space;8&space;&&space;5&space;&&space;7&space;&&space;1&space;&&space;13&space;&&space;23&space;&&space;11&space;&&space;17\\&space;11&space;&&space;7&space;&&space;9&space;&&space;5&space;&&space;3&space;&&space;8&space;&&space;26&space;&&space;13&space;&&space;17\\&space;23&space;&&space;1&space;&&space;5&space;&&space;19&space;&&space;11&space;&&space;7&space;&&space;9&space;&&space;4&space;&&space;16\\&space;31&space;&&space;5&space;&&space;12&space;&&space;7&space;&&space;13&space;&&space;17&space;&&space;24&space;&&space;3&space;&&space;11&space;\end{bmatrix}\begin{bmatrix}&space;x_{1}\\&space;x_{2}\\&space;x_{3}\\&space;x_{4}\\&space;x_{5}\\&space;x_{6}\\&space;x_{7}\\&space;x_{8}\\&space;x_{9}&space;\end{bmatrix}=\begin{bmatrix}&space;2\\&space;5\\&space;7\\&space;1\\&space;6\\&space;9\\&space;4\\&space;8\\&space;3&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;21&space;&&space;32&space;&&space;14&space;&&space;8&space;&&space;6&space;&&space;9&space;&&space;11&space;&&space;3&space;&&space;5\\&space;17&space;&&space;2&space;&&space;8&space;&&space;14&space;&&space;55&space;&&space;23&space;&&space;19&space;&&space;1&space;&&space;6\\&space;41&space;&&space;23&space;&&space;13&space;&&space;5&space;&&space;11&space;&&space;22&space;&&space;26&space;&&space;7&space;&&space;9\\&space;12&space;&&space;11&space;&&space;5&space;&&space;8&space;&&space;3&space;&&space;15&space;&&space;7&space;&&space;25&space;&&space;19\\&space;14&space;&&space;7&space;&&space;3&space;&&space;5&space;&&space;11&space;&&space;23&space;&&space;8&space;&&space;7&space;&&space;9\\&space;2&space;&&space;8&space;&&space;5&space;&&space;7&space;&&space;1&space;&&space;13&space;&&space;23&space;&&space;11&space;&&space;17\\&space;11&space;&&space;7&space;&&space;9&space;&&space;5&space;&&space;3&space;&&space;8&space;&&space;26&space;&&space;13&space;&&space;17\\&space;23&space;&&space;1&space;&&space;5&space;&&space;19&space;&&space;11&space;&&space;7&space;&&space;9&space;&&space;4&space;&&space;16\\&space;31&space;&&space;5&space;&&space;12&space;&&space;7&space;&&space;13&space;&&space;17&space;&&space;24&space;&&space;3&space;&&space;11&space;\end{bmatrix}\begin{bmatrix}&space;x_{1}\\&space;x_{2}\\&space;x_{3}\\&space;x_{4}\\&space;x_{5}\\&space;x_{6}\\&space;x_{7}\\&space;x_{8}\\&space;x_{9}&space;\end{bmatrix}=\begin{bmatrix}&space;2\\&space;5\\&space;7\\&space;1\\&space;6\\&space;9\\&space;4\\&space;8\\&space;3&space;\end{bmatrix}" title="\begin{bmatrix} 21 & 32 & 14 & 8 & 6 & 9 & 11 & 3 & 5\\ 17 & 2 & 8 & 14 & 55 & 23 & 19 & 1 & 6\\ 41 & 23 & 13 & 5 & 11 & 22 & 26 & 7 & 9\\ 12 & 11 & 5 & 8 & 3 & 15 & 7 & 25 & 19\\ 14 & 7 & 3 & 5 & 11 & 23 & 8 & 7 & 9\\ 2 & 8 & 5 & 7 & 1 & 13 & 23 & 11 & 17\\ 11 & 7 & 9 & 5 & 3 & 8 & 26 & 13 & 17\\ 23 & 1 & 5 & 19 & 11 & 7 & 9 & 4 & 16\\ 31 & 5 & 12 & 7 & 13 & 17 & 24 & 3 & 11 \end{bmatrix}\begin{bmatrix} x_{1}\\ x_{2}\\ x_{3}\\ x_{4}\\ x_{5}\\ x_{6}\\ x_{7}\\ x_{8}\\ x_{9} \end{bmatrix}=\begin{bmatrix} 2\\ 5\\ 7\\ 1\\ 6\\ 9\\ 4\\ 8\\ 3 \end{bmatrix}" /></a>

The code below will will comptue the LU decomposition by using elimination matrices to solve the system above:

```python
import numpy as np

#elimination matrix implementation
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
```

The following output was generated:

<img src = "../homework-2/elimLU.png">

## Problem 3

The proof is as shown below:

<img src = "../homework-2/p3_proof.png">

## Problem 4

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;21.0&space;&&space;67.0&space;&&space;88.0&space;&&space;73.0\\&space;76.0&space;&&space;63.0&space;&&space;7.0&space;&&space;20.0\\&space;0.0&space;&&space;85.0&space;&&space;56.0&space;&&space;54.0\\&space;19.3&space;&&space;43.0&space;&&space;30.2&space;&&space;29.4&space;\end{bmatrix}\begin{bmatrix}&space;x_{1}\\&space;x_{2}\\&space;x_{3}\\&space;x_{4}&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;141.0\\&space;109.0\\&space;218.0\\&space;93.7&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;21.0&space;&&space;67.0&space;&&space;88.0&space;&&space;73.0\\&space;76.0&space;&&space;63.0&space;&&space;7.0&space;&&space;20.0\\&space;0.0&space;&&space;85.0&space;&&space;56.0&space;&&space;54.0\\&space;19.3&space;&&space;43.0&space;&&space;30.2&space;&&space;29.4&space;\end{bmatrix}\begin{bmatrix}&space;x_{1}\\&space;x_{2}\\&space;x_{3}\\&space;x_{4}&space;\end{bmatrix}&space;=&space;\begin{bmatrix}&space;141.0\\&space;109.0\\&space;218.0\\&space;93.7&space;\end{bmatrix}" title="\begin{bmatrix} 21.0 & 67.0 & 88.0 & 73.0\\ 76.0 & 63.0 & 7.0 & 20.0\\ 0.0 & 85.0 & 56.0 & 54.0\\ 19.3 & 43.0 & 30.2 & 29.4 \end{bmatrix}\begin{bmatrix} x_{1}\\ x_{2}\\ x_{3}\\ x_{4} \end{bmatrix} = \begin{bmatrix} 141.0\\ 109.0\\ 218.0\\ 93.7 \end{bmatrix}" /></a>

The code below uses a single-precision Gaussian Elimination routine to check how many times the solution to the above systems of equations has improved:

```python
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
```

With a chosen ε = 10e-14, the following output was produced from which one can observe that the error (comptued using the L<sub>∞</sub> norm) decreases over each iteration

<img src = "../homework-2/p4_out.png">

## Problem 5

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;\varepsilon&space;&&space;1\\&space;1&space;&&space;1&space;\end{bmatrix}\begin{bmatrix}&space;x_{1}\\&space;x_{2}&space;\end{bmatrix}=\begin{bmatrix}&space;1&space;&plus;&space;\varepsilon\\&space;2&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;\varepsilon&space;&&space;1\\&space;1&space;&&space;1&space;\end{bmatrix}\begin{bmatrix}&space;x_{1}\\&space;x_{2}&space;\end{bmatrix}=\begin{bmatrix}&space;1&space;&plus;&space;\varepsilon\\&space;2&space;\end{bmatrix}" title="\begin{bmatrix} \varepsilon & 1\\ 1 & 1 \end{bmatrix}\begin{bmatrix} x_{1}\\ x_{2} \end{bmatrix}=\begin{bmatrix} 1 + \varepsilon\\ 2 \end{bmatrix}" /></a>

The below code solves the above system where ε = -10<sup>-2k</sup>, 1 &le; k &le; 10:

```python
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
	print '  epsilon		     error'
	print '------------------------------------------'
	for k in range(1,11):
		eps = math.pow(10, -2*k)
		A = np.matrix([[eps,1],[1,1]])
		b = np.array([(1+eps),2])
		L,U = gauss_elim(A)
		x=np.zeros(2)
		y=np.zeros(2)
		forward_sub(L,b,y)
		back_sub(U,y,x)
		print '  ', eps, '\t\t', la.norm([(x[i]-x_exact[i]) for i in range (0,2)])

if __name__ == "__main__":
	main()
``` 

When LU Factorization *without* pivoting is used to solve the linear system, the solution progressively worsens at around ε = 1e-12, as shown in the output below:

<img src = "../homework-2/p5_out.png">