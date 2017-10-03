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
