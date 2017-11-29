import numpy as np

def add(arg1, arg2): #helper method to add lists
    array = []
    for i in range(len(arg2)): #arg1's size must be <= arg2's size
        if i >= len(arg1):
            array.append(arg2[i])
        else:
            array.append(arg1[i] + arg2[i])
    return array

def flip(arr): #flips array
    return arr[::-1]

def valueOf(coefficients, value):
    n = len(coefficients)
    if n == 1:
        return coefficients[0] #base case
    return coefficients[0] + value * valueOf(coefficients[1:], value)

def derive(coefficients, derivative):
    n = len(coefficients)
    if n == 1: #base case
        return [0] * (len(derivative))
    derivative.append(coefficients[n-1])
    return add(np.copy(derivative), derive(coefficients[:n-1], derivative))

def extraCredit(coefficients, x):
    value = valueOf(coefficients, x)
    derivative = flip(derive(coefficients, []))
    derivativeValue = valueOf(derivative, x)
    return value, derivativeValue, derivative
def main():
    f1 = [2, 3, 5, 4] # f(x) = 2 + 3x + 5x^2 + 4x^3
    x = 3
    v1, dv1, d1 = extraCredit(f1, x)
    print('Value f1 = ', v1)
    print('Derivative f1 = ', d1)
    print('Derivative Value @ x = 3 = ', dv1)

    f2 = [9, -10, 0, -5, 0, 2] # f(x) = 9 - 10x - 5x^3 + 2x^5
    v2, dv2, d2 = extraCredit(f2, x)
    print('Value f2 = ', v2)
    print('Derivative f2 = ', d2)
    print('Derivative Value @ x = 3 = ', dv2)

    f3 = [24, 12, -8, 5, 4] # f(x) = 24 + 12x - 8x^2 + 5x^3 + 4x^4
    v3, dv3, d3 = extraCredit(f3, x)
    print('Value f3 = ', v3)
    print('Derivative f3 = ', d3)
    print('Derivative Value @ x = 3 = ', dv3)
main()