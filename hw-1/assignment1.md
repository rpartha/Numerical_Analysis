# Homework #1, 10/03/17
Ramaseshan Parthasarathy  
Akhil Velagapudi  
Tarun Sreenathan  

## Problem 1

The code to compute absolute and relative error given approximation is as follows:
```python
import numpy as np
import math as math

def computeError(value, approx):
	absolute = np.float32(math.fabs(np.float32(value - approx)))
	relative = np.float32(math.fabs(np.float32(value - approx))/np.float32(math.fabs(value)))

	print("Absolute Error: %s" % (absolute))
	print("Relative Error: %s\n" % (relative))


print("a.")
computeError(np.float32(math.pi), np.float32(3))

print("b.")
computeError(np.float32(math.pi), np.float32(3.14))

print("c.")
computeError(np.float32(math.pi), np.float32(22/7.0))
```

The above code generated the following errors for pi given three different approximations:
```
Using single precision:

a.
Absolute error: 0.141593
Relative error: 0.0450704
b.
Absolute error: 0.00159264
Relative error: 0.000506952
c.
Absolute error: 0.00126433
Relative error: 0.00040245
```

## Problem 2

Machine epislon is stated to be the smallest ε such that 1 - ε < 1 < 1 + ε. It is the _smallest_ quantity representable by the computer and provides spacing between machine representable integers. In a single precision, it can store apprimxately 7 digits (≈ 2^23) while in double precision it can store up to 15 digits (≈ 2^56). During floating-point computations, the machine epsilon forms an upper bound on relative error. Thus, the inequality 1 + ε ≠ 1 holds true. 

## Problem 3
The code to compute the sterling  approximation is as follows:

```python
import numpy as np
import math as math

def power32(a, b):
	return np.float32(a ** b)

# compute the sterling approximation for 
# double and single precision
def approximate_sterling(count):
	#initialize variabales
	fact = 1
	approx = None
	abserr = None
	relerr = None

	print("DOUBLE PRECISION: ")
	for n in range (1, count):
		fact  = fact * n
		root = math.sqrt(2.0 * math.pi * n)
		expo = math.exp(-n)
		power = math.pow(n, n)
		approx = root * expo * power
		abserr = math.fabs(fact-approx)
		relerr = abserr/fact
		print("n = %s, sterling approximation = %s, absolute error = %s, relative error = %s" 
			   % (n, approx, abserr, relerr))

	#re-initialize variables
	fact = 1
	approx = None
	abserr = None
	relerr = None

	#single precision
	print("SINGLE PRECISION: ")
	
	for n in range (1, count):
		fact  = fact * n
		root = np.float32(math.sqrt(np.float32(np.float32(2.0) * np.float32(math.pi)) * np.float32(n)))
		expo = power32(np.float32(math.e), np.float32(-n))
		power = power32(np.float32(n), np.float32(n))
		approx = np.float32(np.float32(root * expo)) * power
		abserr = math.fabs(fact - approx)
		relerr = abserr/fact
		print("n = %s, sterling approximation = %s, absolute error = %s, relative error = %s" 
			   % (n, approx, abserr, relerr))

#execute method for n = 1,2,...,10
approximate_sterling(11)
```

Running the above code would generate an output like so:

<img src = "../hw-1/sterling.png">

From the output, we can see that for both single and double precision, as *n* increases, the absolute error **increases** but the relative error **decreases**. It can be seen that, for some odd reason, that the single precision that uses float32 is more precise than double precision (default to Python3) but remains unchanged. 

## Problem 4 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{\infty}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{1}&space;\leq&space;n&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{\infty}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{\infty}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{1}&space;\leq&space;n&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{\infty}" title="\left \| x \right \| \right \|_{\infty} \leq \left \| x \right \| \right \|_{1} \leq n \cdot \left \| x \right \| \right \|_{\infty}" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{2}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{1}&space;\leq&space;\sqrt{n}&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{2}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{1}&space;\leq&space;\sqrt{n}&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{2}" title="\left \| x \right \| \right \|_{2} \leq \left \| x \right \| \right \|_{1} \leq \sqrt{n} \cdot \left \| x \right \| \right \|_{2}" /></a>



<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{1}&space;\leq&space;\sqrt{n}&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{1}&space;\leq&space;\sqrt{n}&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|&space;\right&space;\|_{2}" title="\left \| x \right \| \right \|_{1} \leq \sqrt{n} \cdot \left \| x \right \| \right \|_{2}" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\frac{\left&space;\|&space;x&space;\right&space;\|_{1}}{\sqrt{n}}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{2},&space;\:&space;n&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\left&space;\|&space;x&space;\right&space;\|_{1}}{\sqrt{n}}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{2},&space;\:&space;n&space;>&space;0" title="\frac{\left \| x \right \|_{1}}{\sqrt{n}} \leq \left \| x \right \|_{2}, \: n > 0" /></a>



<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;\left.&space;\begin{matrix}\frac{\left&space;\|&space;x&space;\right&space;\|_{1}}{\sqrt{n}}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{2}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{1}\\&space;\left&space;\|&space;x&space;\right&space;\|_{\infty}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{1}&space;\end{matrix}&space;\right&space;\}\Rightarrow&space;\left.\begin{matrix}\frac{\left&space;\|&space;x&space;\right&space;\|_{\infty}}{\sqrt{n}}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{2}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{1}\\&space;\left&space;\|&space;x&space;\right&space;\|_{1}&space;\leq&space;n&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|_{\infty}&space;\end{matrix}\right&space;\}\Rightarrow&space;\frac{\left&space;\|&space;x&space;\right&space;\|_{\infty}}{\sqrt{n}}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{2}&space;\leq&space;n&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|_{\infty}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;\left.&space;\begin{matrix}\frac{\left&space;\|&space;x&space;\right&space;\|_{1}}{\sqrt{n}}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{2}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{1}\\&space;\left&space;\|&space;x&space;\right&space;\|_{\infty}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{1}&space;\end{matrix}&space;\right&space;\}\Rightarrow&space;\left.\begin{matrix}\frac{\left&space;\|&space;x&space;\right&space;\|_{\infty}}{\sqrt{n}}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{2}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{1}\\&space;\left&space;\|&space;x&space;\right&space;\|_{1}&space;\leq&space;n&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|_{\infty}&space;\end{matrix}\right&space;\}\Rightarrow&space;\frac{\left&space;\|&space;x&space;\right&space;\|_{\infty}}{\sqrt{n}}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{2}&space;\leq&space;n&space;\cdot&space;\left&space;\|&space;x&space;\right&space;\|_{\infty}" title="\small \left. \begin{matrix}\frac{\left \| x \right \|_{1}}{\sqrt{n}} \leq \left \| x \right \|_{2} \leq \left \| x \right \|_{1}\\ \left \| x \right \|_{\infty} \leq \left \| x \right \|_{1} \end{matrix} \right \}\Rightarrow \left.\begin{matrix}\frac{\left \| x \right \|_{\infty}}{\sqrt{n}} \leq \left \| x \right \|_{2} \leq \left \| x \right \|_{1}\\ \left \| x \right \|_{1} \leq n \cdot \left \| x \right \|_{\infty} \end{matrix}\right \}\Rightarrow \frac{\left \| x \right \|_{\infty}}{\sqrt{n}} \leq \left \| x \right \|_{2} \leq n \cdot \left \| x \right \|_{\infty}" /></a>  

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\texttt{Two&space;vector&space;norms}&space;\left&space;\|&space;x&space;\right&space;\|_{a}&space;\texttt{and&space;}&space;\left&space;\|&space;x&space;\right&space;\|_{b}&space;\texttt{&space;are&space;considered&space;equivalent&space;if&space;there&space;exist&space;real&space;numbers&space;c,d&space;>&space;0&space;such&space;that:}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\texttt{Two&space;vector&space;norms}&space;\left&space;\|&space;x&space;\right&space;\|_{a}&space;\texttt{and&space;}&space;\left&space;\|&space;x&space;\right&space;\|_{b}&space;\texttt{&space;are&space;considered&space;equivalent&space;if&space;there&space;exist&space;real&space;numbers&space;c,d&space;>&space;0&space;such&space;that:}" title="\texttt{Two vector norms} \left \| x \right \|_{a} \texttt{and } \left \| x \right \|_{b} \texttt{ are considered equivalent if there exist real numbers c,d > 0 such that:}" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{120}&space;\small&space;{\color{Red}&space;c\left&space;\|&space;x&space;\right&space;\|_{a}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{b}&space;\leq&space;d\left&space;\|&space;x&space;\right&space;\|_{a}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\small&space;{\color{Red}&space;c\left&space;\|&space;x&space;\right&space;\|_{a}&space;\leq&space;\left&space;\|&space;x&space;\right&space;\|_{b}&space;\leq&space;d\left&space;\|&space;x&space;\right&space;\|_{a}}" title="\small {\color{Red} c\left \| x \right \|_{a} \leq \left \| x \right \|_{b} \leq d\left \| x \right \|_{a}}" /></a>  

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\color{Blue}&space;\therefore&space;\left&space;\|&space;x&space;\right&space;\|_{2}&space;\equiv&space;\left&space;\|&space;x&space;\right&space;\|_{\infty}&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;{\color{Blue}&space;\therefore&space;\left&space;\|&space;x&space;\right&space;\|_{2}&space;\equiv&space;\left&space;\|&space;x&space;\right&space;\|_{\infty}&space;}" title="{\color{Blue} \therefore \left \| x \right \|_{2} \equiv \left \| x \right \|_{\infty} }" /></a>


## Problem 5

The modified code is as follows:

```python
# blur.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse import lil_matrix

# read image file
fname = 'chill.jpg'
image = Image.open(fname).convert("L")
arr = np.asarray(image)
arr.setflags(write = 1)

# initialize blurring matrix
m = arr.shape[0]
n = arr.shape[1]
dofs = m*n
A = lil_matrix((dofs,dofs))
A.setdiag(np.ones(dofs))
for i in range(1,m-1):
    for j in range(1,n-1):
        A[n*i+j,n*i+j] = 8./16.
        A[n*i+j,n*(i-1)+j] = 1./16.
        A[n*i+j,n*(i-1)+(j-1)] = 1./16.
        A[n*i+j,n*(i-1)+(j+1)] = 1./16.
        A[n*i+j,n*i+j-1] = 1./16.
        A[n*i+j,n*i+j+1] = 1./16.
        A[n*i+j,n*(i+1)+j] = 1./16.
        A[n*i+j,n*(i+1)+(j-1)] = 1./16.
        A[n*i+j,n*(i+1)+(j+1)] = 1./16.
A = A.tocsr()

# Blurring function - converts image to a vector, multiplies by
# the blurring matrix, and copies the result back into the image
def blur():
    x = np.zeros(shape=(dofs,1))
    for i in range(0,m):
        for j in range(0,n):
            x[n*i+j] = arr[i,j]

    y = A.dot(x)
    for i in range(0,m):
        for j in range(0,n):
            arr[i,j] = y[n*i+j]

# Execute the blurring function 20 times
for i in range(0,20):
    blur()

# Display the blurred image
plt.imshow(arr,cmap='gray')
plt.show()
```

This code results in the following output image:   
<img src = "../hw-1/blur.png">

As opposed to the original code, which blurred the background but focused on the dog/duck, this code blurs the entire picture. It works in the same way in that the operation performed on the pixel replaces the grayscale value at every pixel by the weighted average of it's neighbors.  