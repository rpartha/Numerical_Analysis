# Homework 3
# Ramaseshan Parthasarathy, Saurabh Prasad

## Problem 1

## Problem 2

## Problem 3

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

We can conclude, based upon the zero in the last column of *A&#42;*, that *A* and *b* are of the same rank. That tells us that the matrix is consistent and has an infinite amount of solutions. We can therefore arrive at the solution by picking an arbitrary *x*<sub>3</sub> and performing backward substitution. The result is as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x&space;=&space;\begin{bmatrix}&space;1/3\\&space;1/3&space;\\&space;0&space;\end{bmatrix}&space;&plus;&space;x_{3}\begin{bmatrix}&space;1\\-2&space;\\1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x&space;=&space;\begin{bmatrix}&space;1/3\\&space;1/3&space;\\&space;0&space;\end{bmatrix}&space;&plus;&space;x_{3}\begin{bmatrix}&space;1\\-2&space;\\1&space;\end{bmatrix}" title="x = \begin{bmatrix} 1/3\\ 1/3 \\ 0 \end{bmatrix} + x_{3}\begin{bmatrix} 1\\-2 \\1 \end{bmatrix}" /></a>

2. Using Gaussian Elimination with partial pivoting using exact arithmetic, the process would fail at the second iteration, at which point there are zero entries in the last row of *A&#42;*. This process was briefly outlined in the first question.

3. The code below computes the solution to the matrix as well as the condition number:

    ```python
        import numpy as np
        from numpy import linalg as linalg

        def main():
            A = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
            b = np.array([0.1,0.3,0.5])

            x= np.linalg.solve(A,b)
            k_cond = np.linalg.cond(A)

            print x
            print 'condition number: ', k_cond

        if __name__ == "__main__":
            main()
    ```

    The output of the above code is as follows:

    <img src = "../hw-3/out.png">

    What's interesting to point out is that the solution does match the description answered in question 1. Our small calculations indicated that the system could have one of infinite solutions. The computed solution gave us one such solution in the form of a 3x1 vector and a condition number of 2.11189683358e+16 was estimated. It should be duely noted that machine epislon is typically on the order of 10<sup>-16</sup> (which can easily be found using np.finfo(float).eps), where as our computation resulted in an order of 10<sup>+16</sup>. 