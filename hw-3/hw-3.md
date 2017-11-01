# Homework 3  

Ramaseshan Parthasarathy, Saurabh Prasad  
11/01/17

## Problem 1

1. The steps to show that the inequality holds is shown below:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\|&space;A&space;\right&space;\|&space;=&space;max_{x\neq&space;0}&space;\frac{\left&space;\|&space;Ax&space;\right&space;\|}{\left&space;\|&space;x&space;\right&space;\|}&space;\geq&space;\frac{\left&space;\|&space;Ay&space;\right&space;\|}{\left&space;\|&space;y&space;\right&space;\|}&space;\Rightarrow&space;\left&space;\|&space;Ay&space;\right&space;\|&space;=&space;\left&space;\|&space;A&space;\right&space;\|\left&space;\|&space;y&space;\right&space;\|,&space;\hspace{1mm}&space;y&space;\in&space;\mathbb{R}^{n}&space;(y&space;\neq&space;0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\left&space;\|&space;A&space;\right&space;\|&space;=&space;max_{x\neq&space;0}&space;\frac{\left&space;\|&space;Ax&space;\right&space;\|}{\left&space;\|&space;x&space;\right&space;\|}&space;\geq&space;\frac{\left&space;\|&space;Ay&space;\right&space;\|}{\left&space;\|&space;y&space;\right&space;\|}&space;\Rightarrow&space;\left&space;\|&space;Ay&space;\right&space;\|&space;=&space;\left&space;\|&space;A&space;\right&space;\|\left&space;\|&space;y&space;\right&space;\|,&space;\hspace{1mm}&space;y&space;\in&space;\mathbb{R}^{n}&space;(y&space;\neq&space;0)" title="\left \| A \right \| = max_{x\neq 0} \frac{\left \| Ax \right \|}{\left \| x \right \|} \geq \frac{\left \| Ay \right \|}{\left \| y \right \|} \Rightarrow \left \| Ay \right \| = \left \| A \right \|\left \| y \right \|, \hspace{1mm} y \in \mathbb{R}^{n} (y \neq 0)" /></a>  

    <a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;\|&space;Ax&space;\right&space;\|&space;\leq&space;\left&space;\|&space;A&space;\right&space;\|\left&space;\|&space;x&space;\right&space;\|&space;\Rightarrow&space;\left&space;\|&space;T^{k}x&space;\right&space;\|&space;\leq&space;\left&space;\|&space;T^{k}&space;\right&space;\|\left&space;\|&space;x&space;\right&space;\|&space;\Rightarrow&space;\left&space;\|&space;T^{k}&space;\right&space;\|&space;\leq&space;\left&space;\|&space;T&space;\right&space;\|^{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;\|&space;Ax&space;\right&space;\|&space;\leq&space;\left&space;\|&space;A&space;\right&space;\|\left&space;\|&space;x&space;\right&space;\|&space;\Rightarrow&space;\left&space;\|&space;T^{k}x&space;\right&space;\|&space;\leq&space;\left&space;\|&space;T^{k}&space;\right&space;\|\left&space;\|&space;x&space;\right&space;\|&space;\Rightarrow&space;\left&space;\|&space;T^{k}&space;\right&space;\|&space;\leq&space;\left&space;\|&space;T&space;\right&space;\|^{k}" title="\left \| Ax \right \| \leq \left \| A \right \|\left \| x \right \| \Rightarrow \left \| T^{k}x \right \| \leq \left \| T^{k} \right \|\left \| x \right \| \Rightarrow \left \| T^{k} \right \| \leq \left \| T \right \|^{k}" /></a>  

    <a href="https://www.codecogs.com/eqnedit.php?latex=\therefore&space;\mathbf{\left&space;\|&space;T^{k}x&space;\right&space;\|&space;\leq&space;\left&space;\|&space;T&space;\right&space;\|^{k}\left&space;\|&space;x&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\therefore&space;\mathbf{\left&space;\|&space;T^{k}x&space;\right&space;\|&space;\leq&space;\left&space;\|&space;T&space;\right&space;\|^{k}\left&space;\|&space;x&space;\right&space;\|}" title="\therefore \mathbf{\left \| T^{k}x \right \| \leq \left \| T \right \|^{k}\left \| x \right \|}" /></a>

2. The proof is rather straight forward:

    <a href="https://www.codecogs.com/eqnedit.php?latex=Ax&space;=&space;b&space;\newline&space;\newline&space;(D-L-U)x&space;\hspace{1mm}&space;=&space;\hspace{1mm}&space;b&space;\newline&space;\newline&space;Dx&space;=&space;(L&plus;U)x&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;b&space;\newline&space;\newline&space;x&space;=&space;D^{-1}(L&plus;U)x&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;D^{-1}b&space;\newline&space;\newline&space;x^{(k)}&space;=&space;D^{-1}(L&plus;U)x^{(k-1)}&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;D^{-1}b&space;\newline&space;\newline&space;x^{(k&plus;1)}&space;=&space;D^{-1}(L&plus;U)x^{(k)}&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;D^{-1}b&space;\newline&space;\newline&space;\mathbf{x^{(k&plus;1)}&space;=&space;Tx^{(k)}&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;c}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Ax&space;=&space;b&space;\newline&space;\newline&space;(D-L-U)x&space;\hspace{1mm}&space;=&space;\hspace{1mm}&space;b&space;\newline&space;\newline&space;Dx&space;=&space;(L&plus;U)x&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;b&space;\newline&space;\newline&space;x&space;=&space;D^{-1}(L&plus;U)x&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;D^{-1}b&space;\newline&space;\newline&space;x^{(k)}&space;=&space;D^{-1}(L&plus;U)x^{(k-1)}&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;D^{-1}b&space;\newline&space;\newline&space;x^{(k&plus;1)}&space;=&space;D^{-1}(L&plus;U)x^{(k)}&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;D^{-1}b&space;\newline&space;\newline&space;\mathbf{x^{(k&plus;1)}&space;=&space;Tx^{(k)}&space;\hspace{1mm}&space;&plus;&space;\hspace{1mm}&space;c}" title="Ax = b \newline \newline (D-L-U)x \hspace{1mm} = \hspace{1mm} b \newline \newline Dx = (L+U)x \hspace{1mm} + \hspace{1mm} b \newline \newline x = D^{-1}(L+U)x \hspace{1mm} + \hspace{1mm} D^{-1}b \newline \newline x^{(k)} = D^{-1}(L+U)x^{(k-1)} \hspace{1mm} + \hspace{1mm} D^{-1}b \newline \newline x^{(k+1)} = D^{-1}(L+U)x^{(k)} \hspace{1mm} + \hspace{1mm} D^{-1}b \newline \newline \mathbf{x^{(k+1)} = Tx^{(k)} \hspace{1mm} + \hspace{1mm} c}" /></a>  

3. Using the previous part, the following can be shown:

    <a href="https://www.codecogs.com/eqnedit.php?latex=x^{*}&space;=&space;Tx^{*}&space;&plus;&space;c&space;\newline&space;\newline&space;x^{(k&plus;1)}&space;\hspace{1mm}&space;-&space;\hspace{1mm}&space;x^{*}&space;=&space;Tx^{(k)}&space;&plus;&space;c&space;-&space;(Tx^{*}&space;&plus;&space;c)&space;\newline&space;\newline&space;e^{(k&plus;1)}&space;=&space;T(x^{(k)}&space;-&space;x^{*})&space;\newline&space;\newline&space;\mathbf{e^{(k&plus;1)}&space;=&space;Te^{(k)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{*}&space;=&space;Tx^{*}&space;&plus;&space;c&space;\newline&space;\newline&space;x^{(k&plus;1)}&space;\hspace{1mm}&space;-&space;\hspace{1mm}&space;x^{*}&space;=&space;Tx^{(k)}&space;&plus;&space;c&space;-&space;(Tx^{*}&space;&plus;&space;c)&space;\newline&space;\newline&space;e^{(k&plus;1)}&space;=&space;T(x^{(k)}&space;-&space;x^{*})&space;\newline&space;\newline&space;\mathbf{e^{(k&plus;1)}&space;=&space;Te^{(k)}}" title="x^{*} = Tx^{*} + c \newline \newline x^{(k+1)} \hspace{1mm} - \hspace{1mm} x^{*} = Tx^{(k)} + c - (Tx^{*} + c) \newline \newline e^{(k+1)} = T(x^{(k)} - x^{*}) \newline \newline \mathbf{e^{(k+1)} = Te^{(k)}}" /></a>  

4. Note that the fact that A is *diagonal dominant*, meaning every diagonal entry is greater than the sum of all other entries in its row

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\|&space;T&space;\right&space;\|_{\infty}&space;=&space;\underset{i}{max}&space;\sum_{i&space;\neq&space;j}^{n}&space;\frac{a_{ij}}{a_{ii}}&space;<&space;1&space;\texttt{&space;since&space;}&space;|a_{ii}|&space;>&space;\sum_{i&space;\neq&space;j}^{n}&space;|a_{ij}|&space;\texttt{&space;and&space;}&space;D^{-1}_{ij}&space;=&space;\frac{1}{a_{ii}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\left&space;\|&space;T&space;\right&space;\|_{\infty}&space;=&space;\underset{i}{max}&space;\sum_{i&space;\neq&space;j}^{n}&space;\frac{a_{ij}}{a_{ii}}&space;<&space;1&space;\texttt{&space;since&space;}&space;|a_{ii}|&space;>&space;\sum_{i&space;\neq&space;j}^{n}&space;|a_{ij}|&space;\texttt{&space;and&space;}&space;D^{-1}_{ij}&space;=&space;\frac{1}{a_{ii}}" title="\left \| T \right \|_{\infty} = \underset{i}{max} \sum_{i \neq j}^{n} \frac{a_{ij}}{a_{ii}} < 1 \texttt{ since } |a_{ii}| > \sum_{i \neq j}^{n} |a_{ij}| \texttt{ and } D^{-1}_{ij} = \frac{1}{a_{ii}}" /></a>

5. Combining steps 1 - 4 would arrive at the conclusion that

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\|&space;e^{(k)}&space;\right&space;\|_{\infty}&space;\leq&space;\left&space;\|&space;T&space;\right&space;\|_{\infty}^{k}\left&space;\|&space;e^{(0)}&space;\right&space;\|_{\infty}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\left&space;\|&space;e^{(k)}&space;\right&space;\|_{\infty}&space;\leq&space;\left&space;\|&space;T&space;\right&space;\|_{\infty}^{k}\left&space;\|&space;e^{(0)}&space;\right&space;\|_{\infty}" title="\left \| e^{(k)} \right \|_{\infty} \leq \left \| T \right \|_{\infty}^{k}\left \| e^{(0)} \right \|_{\infty}" /></a>

## Problem 2

```python
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
```
The output plot(s) of the above program have been provided for convenience:

<img src = "../hw-3/plot_given.png">  
<img src = "../hw-3/plot0.png">  
<img src = "../hw-3/plot1.png">  
<img src = "../hw-3/plot2.png">  
<img src = "../hw-3/plot3.png">  
<img src = "../hw-3/plot4.png">  
<img src = "../hw-3/plot5.png">  

From the plots, it can, of course, be seen that polynomial with degree n = 5 displays the best trend for the data. 


## Problem 3

```python
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
```

The solutions to the systems in part a and b have been given below:

<img src = "../hw-3/outp3.png">

## Problem 4

1. The matrix *A* in question is shown below. We can prove that *A* is singular by proving that *A* is not invertible since singular matrices, by definition, do not have inverses.  

    <a href="https://www.codecogs.com/eqnedit.php?latex=A&space;=&space;\begin{bmatrix}&space;0.1&space;&&space;0.2&space;&&space;0.3\\&space;0.4&space;&&space;0.5&space;&&space;0.6\\&space;0.7&space;&&space;0.8&space;&&space;0.9&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A&space;=&space;\begin{bmatrix}&space;0.1&space;&&space;0.2&space;&&space;0.3\\&space;0.4&space;&&space;0.5&space;&&space;0.6\\&space;0.7&space;&&space;0.8&space;&&space;0.9&space;\end{bmatrix}" title="A = \begin{bmatrix} 0.1 & 0.2 & 0.3\\ 0.4 & 0.5 & 0.6\\ 0.7 & 0.8 & 0.9 \end{bmatrix}" /></a>  

    The determinant can be computed by expanding from the first row:  

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;0.1(0.45-0.48)&space;-&space;0.2(0.36&space;-&space;0.42)&space;&plus;&space;0.3(0.32&space;-&space;0.45)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;0.1(0.45-0.48)&space;-&space;0.2(0.36&space;-&space;0.42)&space;&plus;&space;0.3(0.32&space;-&space;0.45)&space;=&space;0" title="0.1(0.45-0.48) - 0.2(0.36 - 0.42) + 0.3(0.32 - 0.45) = 0" /></a>  

    Knowing that *A* is singular, we can attempt to solve the system *Ax* = *b* below by first setting up the augmented matrix *A&#42;*:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;A^{*}&space;=&space;[A|b]&space;=&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0.4&space;&&space;0.5&space;&&space;0.6&space;\hspace{1mm}&space;\\&space;0.7&space;&&space;0.8&space;&&space;0.9&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;0.3\\&space;0.5&space;\end{matrix}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;A^{*}&space;=&space;[A|b]&space;=&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0.4&space;&&space;0.5&space;&&space;0.6&space;\hspace{1mm}&space;\\&space;0.7&space;&&space;0.8&space;&&space;0.9&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;0.3\\&space;0.5&space;\end{matrix}&space;\end{bmatrix}" title="A^{*} = [A|b] = \begin{bmatrix} \left.\begin{matrix} 0.1 & 0.2 & 0.3 \hspace{1mm} \\ 0.4 & 0.5 & 0.6 \hspace{1mm} \\ 0.7 & 0.8 & 0.9 \hspace{1mm} \end{matrix}\right|\begin{matrix} 0.1 \\ 0.3\\ 0.5 \end{matrix} \end{bmatrix}" /></a> 

    The following steps outline the row operations that can be performed on this matrix:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(1)&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0.4&space;&&space;0.5&space;&&space;0.6&space;\hspace{1mm}&space;\\&space;0.7&space;&&space;0.8&space;&&space;0.9&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;0.3\\&space;0.5&space;\end{matrix}&space;\end{bmatrix}&space;\xrightarrow[R_{3}&space;\hspace{1mm}&space;\rightarrow&space;\hspace{1mm}&space;R_{3}&space;\hspace{1mm}&space;-&space;\hspace{1mm}7R_{1}]{R_{2}&space;\hspace{1mm}&space;\rightarrow&space;\hspace{1mm}&space;R_{2}&space;\hspace{1mm}&space;-&space;\hspace{1mm}4R_{1}}&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.3&space;&&space;-0.6&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.6&space;&&space;-1.2&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;-0.1\\&space;-0.2&space;\end{matrix}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(1)&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0.4&space;&&space;0.5&space;&&space;0.6&space;\hspace{1mm}&space;\\&space;0.7&space;&&space;0.8&space;&&space;0.9&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;0.3\\&space;0.5&space;\end{matrix}&space;\end{bmatrix}&space;\xrightarrow[R_{3}&space;\hspace{1mm}&space;\rightarrow&space;\hspace{1mm}&space;R_{3}&space;\hspace{1mm}&space;-&space;\hspace{1mm}7R_{1}]{R_{2}&space;\hspace{1mm}&space;\rightarrow&space;\hspace{1mm}&space;R_{2}&space;\hspace{1mm}&space;-&space;\hspace{1mm}4R_{1}}&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.3&space;&&space;-0.6&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.6&space;&&space;-1.2&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;-0.1\\&space;-0.2&space;\end{matrix}&space;\end{bmatrix}" title="(1) \begin{bmatrix} \left.\begin{matrix} 0.1 & 0.2 & 0.3 \hspace{1mm} \\ 0.4 & 0.5 & 0.6 \hspace{1mm} \\ 0.7 & 0.8 & 0.9 \hspace{1mm} \end{matrix}\right|\begin{matrix} 0.1 \\ 0.3\\ 0.5 \end{matrix} \end{bmatrix} \xrightarrow[R_{3} \hspace{1mm} \rightarrow \hspace{1mm} R_{3} \hspace{1mm} - \hspace{1mm}7R_{1}]{R_{2} \hspace{1mm} \rightarrow \hspace{1mm} R_{2} \hspace{1mm} - \hspace{1mm}4R_{1}} \begin{bmatrix} \left.\begin{matrix} 0.1 & 0.2 & 0.3 \hspace{1mm} \\ 0 & -0.3 & -0.6 \hspace{1mm} \\ 0 & -0.6 & -1.2 \hspace{1mm} \end{matrix}\right|\begin{matrix} 0.1 \\ -0.1\\ -0.2 \end{matrix} \end{bmatrix}" /></a>  

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(2)&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.3&space;&&space;-0.6&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.6&space;&&space;-1.2&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;-0.1\\&space;-0.2&space;\end{matrix}&space;\end{bmatrix}&space;\xrightarrow[]{R_{3}&space;\hspace{1mm}&space;\rightarrow&space;\hspace{1mm}&space;R_{3}&space;\hspace{1mm}&space;-&space;\hspace{1mm}2R_{2}}&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.3&space;&&space;-0.6&space;\hspace{1mm}&space;\\&space;0&space;&&space;0&space;&&space;0&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;-0.1\\&space;0&space;\end{matrix}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(2)&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.3&space;&&space;-0.6&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.6&space;&&space;-1.2&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;-0.1\\&space;-0.2&space;\end{matrix}&space;\end{bmatrix}&space;\xrightarrow[]{R_{3}&space;\hspace{1mm}&space;\rightarrow&space;\hspace{1mm}&space;R_{3}&space;\hspace{1mm}&space;-&space;\hspace{1mm}2R_{2}}&space;\begin{bmatrix}&space;\left.\begin{matrix}&space;0.1&space;&&space;0.2&space;&&space;0.3&space;\hspace{1mm}&space;\\&space;0&space;&&space;-0.3&space;&&space;-0.6&space;\hspace{1mm}&space;\\&space;0&space;&&space;0&space;&&space;0&space;\hspace{1mm}&space;\end{matrix}\right|\begin{matrix}&space;0.1&space;\\&space;-0.1\\&space;0&space;\end{matrix}&space;\end{bmatrix}" title="(2) \begin{bmatrix} \left.\begin{matrix} 0.1 & 0.2 & 0.3 \hspace{1mm} \\ 0 & -0.3 & -0.6 \hspace{1mm} \\ 0 & -0.6 & -1.2 \hspace{1mm} \end{matrix}\right|\begin{matrix} 0.1 \\ -0.1\\ -0.2 \end{matrix} \end{bmatrix} \xrightarrow[]{R_{3} \hspace{1mm} \rightarrow \hspace{1mm} R_{3} \hspace{1mm} - \hspace{1mm}2R_{2}} \begin{bmatrix} \left.\begin{matrix} 0.1 & 0.2 & 0.3 \hspace{1mm} \\ 0 & -0.3 & -0.6 \hspace{1mm} \\ 0 & 0 & 0 \hspace{1mm} \end{matrix}\right|\begin{matrix} 0.1 \\ -0.1\\ 0 \end{matrix} \end{bmatrix}" /></a>

    We can conclude, based upon the zero in the last column of *A&#42;*, that *A* and *b* are of the same rank. That tells us that the matrix is consistent and has an infinite amount of solutions. We can therefore arrive at a solution by picking an arbitrary *x*<sub>3</sub> and performing backward substitution. The result is as follows:

    <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x&space;=&space;\begin{bmatrix}&space;1/3\\&space;1/3&space;\\&space;0&space;\end{bmatrix}&space;&plus;&space;x_{3}\begin{bmatrix}&space;1\\-2&space;\\1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x&space;=&space;\begin{bmatrix}&space;1/3\\&space;1/3&space;\\&space;0&space;\end{bmatrix}&space;&plus;&space;x_{3}\begin{bmatrix}&space;1\\-2&space;\\1&space;\end{bmatrix}" title="x = \begin{bmatrix} 1/3\\ 1/3 \\ 0 \end{bmatrix} + x_{3}\begin{bmatrix} 1\\-2 \\1 \end{bmatrix}" /></a>

2. Using Gaussian Elimination with partial pivoting using exact arithmetic, the process would fail at the second iteration, at which point there are zero entries in the last row of *A&#42;*. This process was briefly outlined in the previous part.

3. The code below computes the solution to the matrix as well as the condition number:

    ```python
    import numpy as np
    from numpy import linalg as linalg

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
            #if A[j,j] == 0:
            #	print 'Matrix is singular!'
            #	return          # matrix is singular
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
            #if A[j,j] == 0:
            #	print 'Matrix is singular!'
            #	return          # matrix is singular
            x[j] = b[j]/A[j,j]
            for i in range(0,j):
                b[i] = b[i] - A[i,j]*x[j]


    def main():
        A = np.matrix([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
        A_inv = np.linalg.inv(A)
        b = np.array([0.1,0.3,0.5])

        k_cond = np.linalg.norm(A, np.inf) * np.linalg.norm(A_inv, np.inf)

        x = np.zeros(3)
        y = np.zeros(3)

        L,U = gauss_elim(A)
        forward_sub(L,b,y)
        back_sub(U,y,x)

        print (x)
        print 'condition number: ', k_cond
        print(np.finfo(float).eps)

    if __name__ == "__main__":
        main()

    ```

    The output of the above code is as follows:

    <img src = "../hw-3/out.png">

    What's interesting to point out is that the solution does match the description answered in part 1. Our small calculations indicated that the system could have one of infinite solutions. The computed solution gave us one such solution in the form of a 3x1 vector and a condition number of 2.11189683358e+16 was estimated. It should be duely noted that machine epislon is typically on the order of 10<sup>-16</sup> (which can easily be found using np.finfo(float).eps), where as our computation resulted in an order of 10<sup>+16</sup>. 