import numpy as np
import math as math


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
		approx = math.sqrt(2.0 * math.pi * n) * math.exp(-n) * math.pow(n, n)
		abserr = math.fabs(fact - approx)
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
		approx = np.float32(math.sqrt(2.0 * math.pi * n) * math.exp(-n) * math.pow(n, n))
		abserr = np.float32(math.fabs(fact - approx))
		relerr = np.float32(abserr/fact)
		print("n = %s, sterling approximation = %s, absolute error = %s, relative error = %s" 
			   % (n, approx, abserr, relerr))

#execute method for n = 1,2,...,10
approximate_sterling(11) 