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