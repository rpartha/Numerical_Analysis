import numpy as np
import math as math

def power32(a, b):
	if(b == 0): return np.float16(1)
	elif(b == 1): return np.float16(1)
	else: return np.float16(a * power(a, b-1))

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
		root = np.float16(math.sqrt(np.float16(np.float16(2.0) * np.float16(math.pi)) * np.float16(n))
		expo = power32(np.float16(math.e), np.float16(-n))
		power = power32(np.float16(n), np.float16(n))
		approx = np.float16(np.float16(root * expo)) * power)
		abserr = math.fabs(fact - approx)
		relerr = abserr/fact
		print("n = %s, sterling approximation = %s, absolute error = %s, relative error = %s" 
			   % (n, approx, abserr, relerr)

#execute method for n = 1,2,...,10
approximate_sterling(11)
