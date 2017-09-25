# Homework #1
Akhil Velagapudi (av481)  
Tarun Sreenathan (ts)  
Ramaseshan Parthasarathy(rp770)  

## Problem 1

Akhil

## Problem 2

Tarun

## Problem 3
The code to compute the sterling  approximation is as follows:

```python
import numpy as np
import math as math

def power32(a, b):
	if(b == 0): return np.float32(1)
	elif(b == 1): return np.float32(1)
	else: return np.float32(a * power(a, b-1))

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
			   % (n, approx, abserr, relerr)

	#re-initialize variables
	fact = 1
	approx = None
	abserr = None
	relerr = None

	#single precision
	print("SINGLE PRECISION: ")
	
	for n in range (1, count):
		fact  = fact * n
		root = np.float32(math.sqrt(np.float32(np.float32(2.0) * np.float32(math.pi)) * np.float32(n))
		expo = power32(np.float32(math.e), np.float32(-n))
		power = power32(np.float32(n), np.float32(n))
		approx = np.float32(np.float32(root * expo)) * power)
		abserr = math.fabs(fact - approx)
		relerr = abserr/fact
		print("n = %s, sterling approximation = %s, absolute error = %s, relative error = %s" 
			   % (n, approx, abserr, relerr)

#execute method for n = 1,2,...,10
approximate_sterling(11)
```

Running the above code would generate an output like so:

<img src = "../Homework_One/sterling.png">

From the output, we can see that for both single and double precision, as *n* increases, the absolute error **increases** but the relative error **decreases**. It can be seen that, for some odd reason, that the single precision that uses float32 is more precise than double precision (default to Python3) but remains unchanged. 

## Problem 4 

Akhil

## Problem 5

Tarun