import numpy as np
import math as math

#return max of sum of columns
def computeL1norm(A):
    column_sums = [sum([math.fabs(row[i]) for row in A]) for i in range(0,len(A[0]))]
    max = column_sums[0]
    for i in range(len(column_sums)):
        if(i > max):
            max = i
    return max

def main():
    A1 = np.array([[10.,-7.,0.],[5.,-1.,5.],[-3.,2.,6.]])
    A2 = np.array([[92.,66.,25.],[-73.,78.,24.],[-80.,37.,10.]])
    normA1 = computeL1norm(A1)
    normA2 = computeL1norm(A2)

    y1A1 = np.random.rand(3,1)
    y2A1 = np.random.rand(3,1)
    y3A1 = np.random.rand(3,1)
    y4A1 = np.random.rand(3,1)
    y5A1 = np.random.rand(3,1)

    z1A1 = np.linalg.solve(A1,y1A1) 
    z2A1 = np.linalg.solve(A1,y2A1) 
    z3A1 = np.linalg.solve(A1,y3A1) 
    z4A1 = np.linalg.solve(A1,y4A1) 
    z5A1 = np.linalg.solve(A1,y5A1) 

    r1A1 = np.linalg.norm(z1A1,1)/np.linalg.norm(y1A1,1)
    r2A1 = np.linalg.norm(z2A1,1)/np.linalg.norm(y2A1,1)
    r3A1 = np.linalg.norm(z3A1,1)/np.linalg.norm(y2A1,1)
    r4A1 = np.linalg.norm(z4A1,1)/np.linalg.norm(y4A1,1)
    r5A1 = np.linalg.norm(z5A1,1)/np.linalg.norm(y5A1,1)

    y1A2 = np.random.rand(3,1)
    y2A2 = np.random.rand(3,1)
    y3A2 = np.random.rand(3,1)
    y4A2 = np.random.rand(3,1)
    y5A2 = np.random.rand(3,1)

    z1A2 = np.linalg.solve(A2,y1A2) 
    z2A2 = np.linalg.solve(A2,y2A2) 
    z3A2 = np.linalg.solve(A2,y3A2) 
    z4A2 = np.linalg.solve(A2,y4A2) 
    z5A2 = np.linalg.solve(A2,y5A2) 

    r1A2 = np.linalg.norm(z1A2,1)/np.linalg.norm(y1A2,1)
    r2A2 = np.linalg.norm(z2A2,1)/np.linalg.norm(y2A2,1)
    r3A2 = np.linalg.norm(z3A2,1)/np.linalg.norm(y2A2,1)
    r4A2 = np.linalg.norm(z4A2,1)/np.linalg.norm(y4A2,1)
    r5A2 = np.linalg.norm(z5A2,1)/np.linalg.norm(y5A2,1)

    kA1 = max(r1A1,r2A1,r3A1,r4A1,r5A1)*normA1
    kA2 = max(r1A2,r2A2,r3A2,r4A2,r5A2)*normA2

    print('condition number for A1: ', kA1)
    print('condition number for A2: ', kA2)

if __name__ == "__main__":
    main()


