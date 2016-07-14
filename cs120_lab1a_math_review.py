# Databricks notebook source exported at Thu, 14 Jul 2016 00:59:32 UTC

# MAGIC %md
# MAGIC <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png"/> </a> <br/> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. </a>

# COMMAND ----------

# MAGIC %md
# MAGIC ![ML Logo](http://spark-mooc.github.io/web-assets/images/CS190.1x_Banner_300.png)
# MAGIC # Math and Python review
# MAGIC 
# MAGIC This notebook reviews vector and matrix math, the [NumPy](http://www.numpy.org/) Python package, and Python lambda expressions.  Part 1 covers vector and matrix math, and you'll do a few exercises by hand.  In Part 2, you'll learn about NumPy and use `ndarray` objects to solve the math exercises.   Part 3 provides additional information about NumPy and how it relates to array usage in Spark's [MLlib](https://spark.apache.org/mllib/).  Part 4 provides an overview of lambda expressions.
# MAGIC 
# MAGIC To move through the notebook just run each of the cells.  You can run a cell by pressing "shift-enter", which will compute the current cell and advance to the next cell, or by clicking in a cell and pressing "control-enter", which will compute the current cell and remain in that cell.  You should move through the notebook from top to bottom and run all of the cells.  If you skip some cells, later cells might not work as expected.
# MAGIC Note that there are several exercises within this notebook.  You will need to provide solutions for cells that start with: `# TODO: Replace <FILL IN> with appropriate code`.
# MAGIC 
# MAGIC ** This notebook covers: **
# MAGIC * *Part 1:* Math review
# MAGIC * *Part 2:* NumPy
# MAGIC * *Part 3:* Additional NumPy and Spark linear algebra
# MAGIC * *Part 4:* Python lambda expressions
# MAGIC * *Appendix A:* Submitting your exercises to the Autograder

# COMMAND ----------

labVersion = 'cs120x-lab1a-1.0.0'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Math review

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1a) Scalar multiplication: vectors
# MAGIC 
# MAGIC In this exercise, you will calculate the product of a scalar and a vector by hand and enter the result in the code cell below.  Scalar multiplication is straightforward.  The resulting vector equals the product of the scalar, which is a single value, and each item in the original vector.
# MAGIC In the example below, \\( a \\) is the scalar (constant) and \\( \mathbf{v} \\) is the vector.  \\[ a \mathbf{v} = \begin{bmatrix} a v_1 \\\ a v_2 \\\ \vdots \\\ a v_n \end{bmatrix} \\]
# MAGIC 
# MAGIC Calculate the value of \\( \mathbf{x} \\): \\[ \mathbf{x} = 3 \begin{bmatrix} 1 \\\ -2 \\\ 0 \end{bmatrix} \\]
# MAGIC Calculate the value of \\( \mathbf{y} \\): \\[ \mathbf{y} = 2 \begin{bmatrix} 2 \\\ 4 \\\ 8 \end{bmatrix} \\]

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Manually calculate your answer and represent the vector as a list of integers.
# For example, [2, 4, 8].
vectorX = <FILL IN>
vectorY = <FILL IN>

# COMMAND ----------

# TEST Scalar multiplication: vectors (1a)
# Import test library
from databricks_test_helper import Test

Test.assertEqualsHashed(vectorX, 'e460f5b87531a2b60e0f55c31b2e49914f779981',
                        'incorrect value for vectorX')
Test.assertEqualsHashed(vectorY, 'e2d37ff11427dbac7f833a5a7039c0de5a740b1e',
                        'incorrect value for vectorY')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1b) Element-wise multiplication: vectors
# MAGIC 
# MAGIC In this exercise, you will calculate the element-wise multiplication of two vectors by hand and enter the result in the code cell below.  You'll later see that element-wise multiplication is the default method when two NumPy arrays are multiplied together.  Note we won't be performing element-wise multiplication in future labs, but we are introducing it here to distinguish it from other vector operators. It is also a common operation in NumPy, as we will discuss in Part (2b).
# MAGIC 
# MAGIC The element-wise calculation is as follows: \\[ \mathbf{x} \odot \mathbf{y} =  \begin{bmatrix} x_1 y_1 \\\  x_2 y_2 \\\ \vdots \\\ x_n y_n \end{bmatrix} \\]
# MAGIC 
# MAGIC Calculate the value of \\( \mathbf{z} \\): \\[ \mathbf{z} = \begin{bmatrix} 1 \\\  2 \\\ 3 \end{bmatrix} \odot \begin{bmatrix} 4 \\\  5 \\\ 6 \end{bmatrix} \\]

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Manually calculate your answer and represent the vector as a list of integers.
z = <FILL IN>

# COMMAND ----------

# TEST Element-wise multiplication: vectors (1b)
Test.assertEqualsHashed(z, '4b5fe28ee2d274d7e0378bf993e28400f66205c2',
                        'incorrect value for z')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1c) Dot product
# MAGIC 
# MAGIC In this exercise, you will calculate the dot product of two vectors by hand and enter the result in the code cell below.  Note that the dot product is equivalent to performing element-wise multiplication and then summing the result.
# MAGIC 
# MAGIC Below, you'll find the calculation for the dot product of two vectors, where each vector has length \\( n \\): \\[ \mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^n w_i x_i \\]
# MAGIC 
# MAGIC Note that you may also see \\( \mathbf{w} \cdot \mathbf{x} \\) represented as \\( \mathbf{w}^\top \mathbf{x} \\)
# MAGIC 
# MAGIC Calculate the value for \\( c_1 \\) based on the dot product of the following two vectors:
# MAGIC \\[ c_1 = \begin{bmatrix} 1 & -3 \end{bmatrix} \cdot \begin{bmatrix} 4 \\\ 5 \end{bmatrix}\\]
# MAGIC 
# MAGIC Calculate the value for \\( c_2 \\) based on the dot product of the following two vectors:
# MAGIC \\[ c_2 = \begin{bmatrix} 3 & 4 & 5 \end{bmatrix} \cdot \begin{bmatrix} 1 \\\ 2 \\\ 3 \end{bmatrix}\\]

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Manually calculate your answer and set the variables to their appropriate integer values.
c1 = <FILL IN>
c2 = <FILL IN>

# COMMAND ----------

# TEST Dot product (1c)
Test.assertEqualsHashed(c1, '8d7a9046b6a6e21d66409ad0849d6ab8aa51007c', 'incorrect value for c1')
Test.assertEqualsHashed(c2, '887309d048beef83ad3eabf2a79a64a389ab1c9f', 'incorrect value for c2')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1d) Matrix multiplication
# MAGIC 
# MAGIC In this exercise, you will calculate the result of multiplying two matrices together by hand and enter the result in the code cell below.
# MAGIC Refer to the slides for the formula for multiplying two matrices together.
# MAGIC 
# MAGIC First, you'll calculate the value for \\( \mathbf{X} \\).
# MAGIC \\[ \mathbf{X} = \begin{bmatrix} 1 & 2 & 3 \\\ 4 & 5 & 6 \end{bmatrix} \begin{bmatrix} 1 & 2 \\\ 3 & 4 \\\ 5 & 6 \end{bmatrix} \\]
# MAGIC 
# MAGIC Next, you'll perform an outer product and calculate the value for \\( \mathbf{Y} \\).
# MAGIC 
# MAGIC \\[ \mathbf{Y} = \begin{bmatrix} 1 \\\ 2 \\\ 3 \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \\]
# MAGIC 
# MAGIC The resulting matrices should be stored row-wise (see [row-major order](https://en.wikipedia.org/wiki/Row-major_order)). This means that the matrix is organized by rows. For instance, a 2x2 row-wise matrix would be represented as: \\( [[r_1c_1, r_1c_2], [r_2c_1, r_2c_2]] \\) where r stands for row and c stands for column.
# MAGIC 
# MAGIC Note that outer product is just a special case of general matrix multiplication and follows the same rules as normal matrix multiplication.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Represent matrices as lists within lists. For example, [[1,2,3], [4,5,6]] represents a matrix with
# two rows and three columns. Use integer values.
matrixX = <FILL IN>
matrixY = <FILL IN>

# COMMAND ----------

# TEST Matrix multiplication (1d)
Test.assertEqualsHashed(matrixX, 'c2ada2598d8a499e5dfb66f27a24f444483cba13',
                        'incorrect value for matrixX')
Test.assertEqualsHashed(matrixY, 'f985daf651531b7d776523836f3068d4c12e4519',
                        'incorrect value for matrixY')

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Part 2: NumPy

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2a) Scalar multiplication
# MAGIC 
# MAGIC [NumPy](http://docs.scipy.org/doc/numpy/reference/) is a Python library for working with arrays.  NumPy provides abstractions that make it easy to treat these underlying arrays as vectors and matrices.  The library is optimized to be fast and memory efficient, and we'll be using it throughout the course.  The building block for NumPy is the [ndarray](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html), which is a multidimensional array of fixed-size that contains elements of one type (e.g. array of floats).
# MAGIC 
# MAGIC For this exercise, you'll create a `ndarray` consisting of the elements \[1, 2, 3\] and multiply this array by 5.  Use [np.array()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) to create the array.  Note that you can pass a Python list into `np.array()`.  To perform scalar multiplication with an `ndarray` just use `*`.
# MAGIC 
# MAGIC Note that if you create an array from a Python list of integers you will obtain a one-dimensional array, *which is equivalent to a vector for our purposes*.

# COMMAND ----------

# It is convention to import NumPy with the alias np
import numpy as np

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Create a numpy array with the values 1, 2, 3
simpleArray = <FILL IN>
# Perform the scalar product of 5 and the numpy array
timesFive = <FILL IN>
print 'simpleArray\n{0}'.format(simpleArray)
print '\ntimesFive\n{0}'.format(timesFive)

# COMMAND ----------

# TEST Scalar multiplication (2a)
Test.assertTrue(np.all(timesFive == [5, 10, 15]), 'incorrect value for timesFive')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2b) Element-wise multiplication and dot product
# MAGIC 
# MAGIC NumPy arrays support both element-wise multiplication and dot product.  Element-wise multiplication occurs automatically when you use the `*` operator to multiply two `ndarray` objects of the same length.
# MAGIC 
# MAGIC To perform the dot product you can use either [np.dot()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html#numpy.dot) or [np.ndarray.dot()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.dot.html).  For example, if you had NumPy arrays `x` and `y`, you could compute their dot product four ways: `np.dot(x, y)`, `np.dot(y, x)`, `x.dot(y)`, or `y.dot(x)`.
# MAGIC 
# MAGIC For this exercise, multiply the arrays `u` and `v` element-wise and compute their dot product.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Create a ndarray based on a range and step size.
u = np.arange(0, 5, .5)
v = np.arange(5, 10, .5)

elementWise = <FILL IN>
dotProduct = <FILL IN>
print 'u: {0}'.format(u)
print 'v: {0}'.format(v)
print '\nelementWise\n{0}'.format(elementWise)
print '\ndotProduct\n{0}'.format(dotProduct)

# COMMAND ----------

# TEST Element-wise multiplication and dot product (2b)
Test.assertTrue(np.all(elementWise == [ 0., 2.75, 6., 9.75, 14., 18.75, 24., 29.75, 36., 42.75]),
                'incorrect value for elementWise')
Test.assertEquals(dotProduct, 183.75, 'incorrect value for dotProduct')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2c) Matrix math
# MAGIC With NumPy it is very easy to perform matrix math.  You can use [np.matrix()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html) to generate a NumPy matrix.  Just pass a two-dimensional `ndarray` or a list of lists to the function.  You can perform matrix math on NumPy matrices using `*`.
# MAGIC 
# MAGIC You can transpose a matrix by calling [numpy.matrix.transpose()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.transpose.html) or by using `.T` on the matrix object (e.g. `myMatrix.T`).  Transposing a matrix produces a matrix where the new rows are the columns from the old matrix. For example: \\[  \begin{bmatrix} 1 & 2 & 3 \\\ 4 & 5 & 6 \end{bmatrix}^\top = \begin{bmatrix} 1 & 4 \\\ 2 & 5 \\\ 3 & 6 \end{bmatrix} \\]
# MAGIC 
# MAGIC Inverting a matrix can be done using [numpy.linalg.inv()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html).  Note that only square matrices can be inverted, and square matrices are not guaranteed to have an inverse.  If the inverse exists, then multiplying a matrix by its inverse will produce the identity matrix.  \\( \scriptsize ( A^{-1} A = I_n ) \\)  The identity matrix \\( \scriptsize I_n \\) has ones along its diagonal and zeros elsewhere. \\[ I_n = \begin{bmatrix} 1 & 0 & 0 & ... & 0 \\\ 0 & 1 & 0 & ... & 0 \\\ 0 & 0 & 1 & ... & 0 \\\ ... & ... & ... & ... & ... \\\ 0 & 0 & 0 & ... & 1 \end{bmatrix} \\]
# MAGIC 
# MAGIC For this exercise, multiply \\( A \\) times its transpose \\( ( A^\top ) \\) and then calculate the inverse of the result \\( (  [ A A^\top ]^{-1}  ) \\).

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
from numpy.linalg import inv

A = np.matrix([[1,2,3,4],[5,6,7,8]])
print 'A:\n{0}'.format(A)
# Print A transpose
print '\nA transpose:\n{0}'.format(A.T)

# Multiply A by A transpose
AAt = <FILL IN>
print '\nAAt:\n{0}'.format(AAt)

# Invert AAt with np.linalg.inv()
AAtInv = <FILL IN>
print '\nAAtInv:\n{0}'.format(AAtInv)

# Show inverse times matrix equals identity
# We round due to numerical precision
print '\nAAtInv * AAt:\n{0}'.format((AAtInv * AAt).round(4))

# COMMAND ----------

# TEST Matrix math (2c)
Test.assertTrue(np.all(AAt == np.matrix([[30, 70], [70, 174]])), 'incorrect value for AAt')
Test.assertTrue(np.allclose(AAtInv, np.matrix([[0.54375, -0.21875], [-0.21875, 0.09375]])),
                'incorrect value for AAtInv')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 3: Additional NumPy and Spark linear algebra

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3a) Slices
# MAGIC 
# MAGIC You can select a subset of a one-dimensional NumPy `ndarray`'s elements by using slices.  These slices operate the same way as slices for Python lists.  For example, `[0, 1, 2, 3][:2]` returns the first two elements `[0, 1]`.  NumPy, additionally, has more sophisticated slicing that allows slicing across multiple dimensions; however, you'll only need to use basic slices in future labs for this course.
# MAGIC 
# MAGIC Note that if no index is placed to the left of a `:`, it is equivalent to starting at 0, and hence `[0, 1, 2, 3][:2]` and `[0, 1, 2, 3][0:2]` yield the same result.  Similarly, if no index is placed to the right of a `:`, it is equivalent to slicing to the end of the object.  Also, you can use negative indices to index relative to the end of the object, so `[-2:]` would return the last two elements of the object.
# MAGIC 
# MAGIC For this exercise, return the last 3 elements of the array `features`.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
features = np.array([1, 2, 3, 4])
print 'features:\n{0}'.format(features)

# The last three elements of features
lastThree = <FILL IN>

print '\nlastThree:\n{0}'.format(lastThree)

# COMMAND ----------

# TEST Slices (3a)
Test.assertTrue(np.all(lastThree == [2, 3, 4]), 'incorrect value for lastThree')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3b) Combining `ndarray` objects
# MAGIC 
# MAGIC NumPy provides many functions for creating new arrays from existing arrays.  We'll explore two functions: [np.hstack()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html), which allows you to combine arrays column-wise, and [np.vstack()](http://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html), which allows you to combine arrays row-wise.  Note that both `np.hstack()` and `np.vstack()` take in a tuple of arrays as their first argument.  To horizontally combine three arrays `a`, `b`, and `c`, you would run `np.hstack((a, b, c))`.
# MAGIC If we had two arrays: `a = [1, 2, 3, 4]` and `b = [5, 6, 7, 8]`, we could use `np.vstack((a, b))` to produce the two-dimensional array: \\[  \begin{bmatrix} 1 & 2 & 3 & 4 \\\ 5 & 6 & 7 & 8 \end{bmatrix} \\]
# MAGIC 
# MAGIC For this exercise, you'll combine the `zeros` and `ones` arrays both horizontally (column-wise) and vertically (row-wise).
# MAGIC Note that the result of stacking two arrays is an `ndarray`.  If you need the result to be a matrix, you can call `np.matrix()` on the result, which will return a NumPy matrix.

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
zeros = np.zeros(8)
ones = np.ones(8)
print 'zeros:\n{0}'.format(zeros)
print '\nones:\n{0}'.format(ones)

zerosThenOnes = <FILL IN>   # A 1 by 16 array
zerosAboveOnes = <FILL IN>  # A 2 by 8 array

print '\nzerosThenOnes:\n{0}'.format(zerosThenOnes)
print '\nzerosAboveOnes:\n{0}'.format(zerosAboveOnes)

# COMMAND ----------

# TEST Combining ndarray objects (3b)
Test.assertTrue(np.all(zerosThenOnes == [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]),
                'incorrect value for zerosThenOnes')
Test.assertTrue(np.all(zerosAboveOnes == [[0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1]]),
                'incorrect value for zerosAboveOnes')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (3c) PySpark's DenseVector
# MAGIC 
# MAGIC PySpark provides a [DenseVector](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.DenseVector) class within the module [pyspark.mllib.linalg](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#module-pyspark.mllib.linalg).  `DenseVector` is used to store arrays of values for use in PySpark.  `DenseVector` actually stores values in a NumPy array and delegates calculations to that object.  You can create a new `DenseVector` using `DenseVector()` and passing in an NumPy array or a Python list.
# MAGIC 
# MAGIC `DenseVector` implements several functions.  The only function needed for this course is `DenseVector.dot()`, which operates just like `np.ndarray.dot()`.
# MAGIC Note that `DenseVector` stores all values as `np.float64`, so even if you pass in an NumPy array of integers, the resulting `DenseVector` will contain floating-point numbers. Also, `DenseVector` objects exist locally and are not inherently distributed.  `DenseVector` objects can be used in the distributed setting by either passing functions that contain them to resilient distributed dataset (RDD) transformations or by distributing them directly as RDDs.
# MAGIC 
# MAGIC For this exercise, create a `DenseVector` consisting of the values `[3.0, 4.0, 5.0]` and compute the dot product of this vector with `numpyVector`.

# COMMAND ----------

from pyspark.mllib.linalg import DenseVector

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
numpyVector = np.array([-3, -4, 5])
print '\nnumpyVector:\n{0}'.format(numpyVector)

# Create a DenseVector consisting of the values [3.0, 4.0, 5.0]
myDenseVector = <FILL IN>
# Calculate the dot product between the two vectors.
denseDotProduct = <FILL IN>

print 'myDenseVector:\n{0}'.format(myDenseVector)
print '\ndenseDotProduct:\n{0}'.format(denseDotProduct)

# COMMAND ----------

# TEST PySpark's DenseVector (3c)
Test.assertTrue(isinstance(myDenseVector, DenseVector), 'myDenseVector is not a DenseVector')
Test.assertTrue(np.allclose(myDenseVector, np.array([3., 4., 5.])),
                'incorrect value for myDenseVector')
Test.assertTrue(np.allclose(denseDotProduct, 0.0), 'incorrect value for denseDotProduct')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Python lambda expressions

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4a) Lambda is an anonymous function
# MAGIC 
# MAGIC We can use a lambda expression to create a function.  To do this, you type `lambda` followed by the names of the function's parameters separated by commas, followed by a `:`, and then the expression statement that the function will evaluate.  For example, `lambda x, y: x + y` is an anonymous function that computes the sum of its two inputs.
# MAGIC 
# MAGIC Lambda expressions return a function when evaluated.  The function is not bound to any variable, which is why lambdas are associated with anonymous functions.  However, it is possible to assign the function to a variable.  Lambda expressions are particularly useful when you need to pass a simple function into another function.  In that case, the lambda expression generates a function that is bound to the parameter being passed into the function.
# MAGIC 
# MAGIC Below, we'll see an example of how we can bind the function returned by a lambda expression to a variable named `addSLambda`.  From this example, we can see that `lambda` provides a shortcut for creating a simple function.  Note that the behavior of the function created using `def` and the function created using `lambda` is equivalent.  Both functions have the same type and return the same results.  The only differences are the names and the way they were created.
# MAGIC For this exercise, first run the two cells below to compare a function created using `def` with a corresponding anonymous function.  Next, write your own lambda expression that creates a function that multiplies its input (a single parameter) by 10.
# MAGIC 
# MAGIC Here are some additional references that explain lambdas: [Lambda Functions](http://www.secnetix.de/olli/Python/lambda_functions.hawk), [Lambda Tutorial](https://pythonconquerstheuniverse.wordpress.com/2011/08/29/lambda_tutorial/), and [Python Functions](http://www.bogotobogo.com/python/python_functions_lambda.php).

# COMMAND ----------

# Example function
def addS(x):
    return x + 's'
print type(addS)
print addS
print addS('cat')

# COMMAND ----------

# As a lambda
addSLambda = lambda x: x + 's'
print type(addSLambda)
print addSLambda
print addSLambda('cat')

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Recall that: "lambda x, y: x + y" creates a function that adds together two numbers
multiplyByTen = lambda x: <FILL IN>
print multiplyByTen(5)

# Note that the function still shows its name as <lambda>
print '\n', multiplyByTen

# COMMAND ----------

# TEST Python lambda expressions (4a)
Test.assertEquals(multiplyByTen(10), 100, 'incorrect definition for multiplyByTen')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4b) `lambda` fewer steps than `def`
# MAGIC 
# MAGIC `lambda` generates a function and returns it, while `def` generates a function and assigns it to a name.  The function returned by `lambda` also automatically returns the value of its expression statement, which reduces the amount of code that needs to be written.
# MAGIC 
# MAGIC For this exercise, recreate the `def` behavior using `lambda`.  Note that since a lambda expression returns a function, it can be used anywhere an object is expected. For example, you can create a list of functions where each function in the list was generated by a lambda expression.

# COMMAND ----------

# Code using def that we will recreate with lambdas
def plus(x, y):
    return x + y

def minus(x, y):
    return x - y

functions = [plus, minus]
print functions[0](4, 5)
print functions[1](4, 5)

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# The first function should add two values, while the second function should subtract the second
# value from the first value.
lambdaFunctions = [lambda <FILL IN> ,  lambda <FILL IN>]
print lambdaFunctions[0](4, 5)
print lambdaFunctions[1](4, 5)

# COMMAND ----------

# TEST lambda fewer steps than def (4b)
Test.assertEquals(lambdaFunctions[0](10, 10), 20, 'incorrect first lambdaFunction')
Test.assertEquals(lambdaFunctions[1](10, 10), 0, 'incorrect second lambdaFunction')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4c) Lambda expression arguments
# MAGIC 
# MAGIC Lambda expressions can be used to generate functions that take in zero or more parameters.  The syntax for `lambda` allows for multiple ways to define the same function.  For example, we might want to create a function that takes in a single parameter, where the parameter is a tuple consisting of two values, and the function adds the two values together.  The syntax could be either: `lambda x: x[0] + x[1]` or `lambda (x0, x1): x0 + x1`.  If we called either function on the tuple `(3, 4)` it would return `7`.  Note that the second `lambda` relies on the tuple `(3, 4)` being unpacked automatically, which means that `x0` is assigned the value `3` and `x1` is assigned the value `4`.
# MAGIC 
# MAGIC As an other example, consider the following parameter lambda expressions: `lambda x, y: (x[0] + y[0], x[1] + y[1])` and `lambda (x0, x1), (y0, y1): (x0 + y0, x1 + y1)`.  The result of applying either of these functions to tuples  `(1, 2)` and `(3, 4)` would be the tuple `(4, 6)`.
# MAGIC 
# MAGIC For this exercise: you'll create one-parameter functions `swap1` and `swap2` that swap the order of a tuple; a one-parameter function `swapOrder` that takes in a tuple with three values and changes the order to: second element, third element, first element; and finally, a three-parameter function `sumThree` that takes in three tuples, each with two values, and returns a tuple containing two values: the sum of the first element of each tuple and the sum of second element of each tuple.

# COMMAND ----------

# Examples.  Note that the spacing has been modified to distinguish parameters from tuples.

# One-parameter function
a1 = lambda x: x[0] + x[1]
a2 = lambda (x0, x1): x0 + x1
print 'a1( (3,4) ) = {0}'.format( a1( (3,4) ) )
print 'a2( (3,4) ) = {0}'.format( a2( (3,4) ) )

# Two-parameter function
b1 = lambda x, y: (x[0] + y[0], x[1] + y[1])
b2 = lambda (x0, x1), (y0, y1): (x0 + y0, x1 + y1)
print '\nb1( (1,2), (3,4) ) = {0}'.format( b1( (1,2), (3,4) ) )
print 'b2( (1,2), (3,4) ) = {0}'.format( b2( (1,2), (3,4) ) )


# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Use both syntaxes to create a function that takes in a tuple of two values and swaps their order
# E.g. (1, 2) => (2, 1)
swap1 = lambda x: <FILL IN>
swap2 = lambda (x0, x1): <FILL IN>
print 'swap1((1, 2)) = {0}'.format(swap1((1, 2)))
print 'swap2((1, 2)) = {0}'.format(swap2((1, 2)))

# Using either syntax, create a function that takes in a tuple with three values and returns a tuple
# of (2nd value, 3rd value, 1st value).  E.g. (1, 2, 3) => (2, 3, 1)
swapOrder = <FILL IN>
print 'swapOrder((1, 2, 3)) = {0}'.format(swapOrder((1, 2, 3)))

# Using either syntax, create a function that takes in three tuples each with two values.  The
# function should return a tuple with the values in the first position summed and the values in the
# second position summed. E.g. (1, 2), (3, 4), (5, 6) => (1 + 3 + 5, 2 + 4 + 6) => (9, 12)
sumThree = <FILL IN>
print 'sumThree((1, 2), (3, 4), (5, 6)) = {0}'.format(sumThree((1, 2), (3, 4), (5, 6)))

# COMMAND ----------

# TEST Lambda expression arguments (4c)
Test.assertEquals(swap1((1, 2)), (2, 1), 'incorrect definition for swap1')
Test.assertEquals(swap2((1, 2)), (2, 1), 'incorrect definition for swap2')
Test.assertEquals(swapOrder((1, 2, 3)), (2, 3, 1), 'incorrect definition for swapOrder')
Test.assertEquals(sumThree((1, 2), (3, 4), (5, 6)), (9, 12), 'incorrect definition for sumThree')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4d) Restrictions on lambda expressions
# MAGIC 
# MAGIC [Lambda expressions](https://docs.python.org/2/reference/expressions.html#lambda) consist of a single [expression statement](https://docs.python.org/2/reference/simple_stmts.html#expression-statements) and cannot contain other [simple statements](https://docs.python.org/2/reference/simple_stmts.html).  In short, this means that the lambda expression needs to evaluate to a value and exist on a single logical line.  If more complex logic is necessary, use `def` in place of `lambda`.
# MAGIC 
# MAGIC Expression statements evaluate to a value (sometimes that value is None).  Lambda expressions automatically return the value of their expression statement.  In fact, a `return` statement in a `lambda` would raise a `SyntaxError`.
# MAGIC 
# MAGIC  The following Python keywords refer to simple statements that cannot be used in a lambda expression: `assert`, `pass`, `del`, `print`, `return`, `yield`, `raise`, `break`, `continue`, `import`, `global`, and `exec`.  Also, note that assignment statements (`=`) and augmented assignment statements (e.g. `+=`) cannot be used either.

# COMMAND ----------

# Just run this code
# This code will fail with a syntax error, as we can't use print in a lambda expression
import traceback
try:
    exec "lambda x: print x"
except:
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4e) Functional programming
# MAGIC 
# MAGIC The `lambda` examples we have shown so far have been somewhat contrived.  This is because they were created to demonstrate the differences and similarities between `lambda` and `def`.  An excellent use case for lambda expressions is functional programming.  In functional programming, you will often pass functions to other functions as parameters, and `lambda` can be used to reduce the amount of code necessary and to make the code more readable.
# MAGIC Some commonly used functions in functional programming are map, filter, and reduce.  Map transforms a series of elements by applying a function individually to each element in the series.  It then returns the series of transformed elements.  Filter also applies a function individually to each element in a series; however, with filter, this function evaluates to `True` or `False` and only elements that evaluate to `True` are retained.  Finally, reduce operates on pairs of elements in a series.  It applies a function that takes in two values and returns a single value.  Using this function, reduce is able to, iteratively, "reduce" a series to a single value.
# MAGIC 
# MAGIC For this exercise, you'll create three simple `lambda` functions, one each for use in map, filter, and reduce.  The map `lambda` will multiply its input by 5, the filter `lambda` will evaluate to `True` for even numbers, and the reduce `lambda` will add two numbers.
# MAGIC 
# MAGIC > Note:
# MAGIC > * We have created a class called `FunctionalWrapper` so that the syntax for this exercise matches the syntax you'll see in PySpark.
# MAGIC > * Map requires a one parameter function that returns a new value, filter requires a one parameter function that returns `True` or `False`, and reduce requires a two parameter function that combines the two parameters and returns a new value.

# COMMAND ----------

# Create a class to give our examples the same syntax as PySpark
class FunctionalWrapper(object):
    def __init__(self, data):
        self.data = data
    def map(self, function):
        """Call `map` on the items in `data` using the provided `function`"""
        return FunctionalWrapper(map(function, self.data))
    def reduce(self, function):
        """Call `reduce` on the items in `data` using the provided `function`"""
        return reduce(function, self.data)
    def filter(self, function):
        """Call `filter` on the items in `data` using the provided `function`"""
        return FunctionalWrapper(filter(function, self.data))
    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)
    def __getattr__(self, name):  return getattr(self.data, name)
    def __getitem__(self, k):  return self.data.__getitem__(k)
    def __repr__(self):  return 'FunctionalWrapper({0})'.format(repr(self.data))
    def __str__(self):  return 'FunctionalWrapper({0})'.format(str(self.data))

# COMMAND ----------

# Map example

# Create some data
mapData = FunctionalWrapper(range(5))

# Define a function to be applied to each element
f = lambda x: x + 3

# Imperative programming: loop through and create a new object by applying f
mapResult = FunctionalWrapper([])  # Initialize the result
for element in mapData:
    mapResult.append(f(element))  # Apply f and save the new value
print 'Result from for loop: {0}'.format(mapResult)

# Functional programming: use map rather than a for loop
print 'Result from map call: {0}'.format(mapData.map(f))

# Note that the results are the same but that the map function abstracts away the implementation
# and requires less code

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
dataset = FunctionalWrapper(range(10))

# Multiply each element by 5
mapResult = dataset.map(<FILL IN>)
# Keep the even elements
# Note that "x % 2" evaluates to the remainder of x divided by 2
filterResult = dataset.filter(<FILL IN>)
# Sum the elements
reduceResult = dataset.reduce(<FILL IN>)

print 'mapResult: {0}'.format(mapResult)
print '\nfilterResult: {0}'.format(filterResult)
print '\nreduceResult: {0}'.format(reduceResult)

# COMMAND ----------

# TEST Functional programming (4e)
Test.assertEquals(mapResult, FunctionalWrapper([0, 5, 10, 15, 20, 25, 30, 35, 40, 45]),
                  'incorrect value for mapResult')
Test.assertEquals(filterResult, FunctionalWrapper([0, 2, 4, 6, 8]),
                  'incorrect value for filterResult')
Test.assertEquals(reduceResult, 45, 'incorrect value for reduceResult')

# COMMAND ----------

# MAGIC %md
# MAGIC ### (4f) Composability
# MAGIC 
# MAGIC Since our methods for map and filter in the `FunctionalWrapper` class return `FunctionalWrapper` objects, we can compose (or chain) together our function calls.  For example, `dataset.map(f1).filter(f2).reduce(f3)`, where `f1`, `f2`, and `f3` are functions or lambda expressions, first applies a map operation to `dataset`, then filters the result from map, and finally reduces the result from the first two operations.
# MAGIC 
# MAGIC  Note that when we compose (chain) an operation, the output of one operation becomes the input for the next operation, and operations are applied from left to right.  It's likely you've seen chaining used with Python strings.  For example, `'Split this'.lower().split(' ')` first returns a new string object `'split this'` and then `split(' ')` is called on that string to produce `['split', 'this']`.
# MAGIC 
# MAGIC For this exercise, reuse your lambda expressions from (4e) but apply them to `dataset` in the sequence: map, filter, reduce.
# MAGIC 
# MAGIC > Note:
# MAGIC > * Since we are composing the operations our result will be different than in (4e).
# MAGIC > * We can write our operations on separate lines to improve readability.

# COMMAND ----------

# Example of a multi-line expression statement
# Note that placing parentheses around the expression allows it to exist on multiple lines without
# causing a syntax error.
(dataset
 .map(lambda x: x + 2)
 .reduce(lambda x, y: x * y))

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
# Multiply the elements in dataset by five, keep just the even values, and sum those values
finalSum = <FILL IN>
print finalSum

# COMMAND ----------

# TEST Composability (4f)
Test.assertEquals(finalSum, 100, 'incorrect value for finalSum')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix A: Submitting Your Exercises to the Autograder
# MAGIC 
# MAGIC This section guides you through Step 2 of the grading process ("Submit to Autograder").
# MAGIC 
# MAGIC Once you confirm that your lab notebook is passing all tests, you can submit it first to the course autograder and then second to the edX website to receive a grade.
# MAGIC 
# MAGIC ** Note that you can only submit to the course autograder once every 1 minute. **

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2(a): Restart your cluster by clicking on the dropdown next to your cluster name and selecting "Restart Cluster".
# MAGIC 
# MAGIC You can do this step in either notebook, since there is one cluster for your notebooks.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/submit_restart.png" alt="Drawing" />

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2(b): _IN THIS NOTEBOOK_, click on "Run All" to run all of the cells.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/submit_runall.png" alt="Drawing" style="height: 80px"/>
# MAGIC 
# MAGIC This step will take some time.
# MAGIC 
# MAGIC Wait for your cluster to finish running the cells in your lab notebook before proceeding.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2(c): Publish this notebook
# MAGIC 
# MAGIC Publish _this_ notebook by clicking on the "Publish" button at the top.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Publish0.png" alt="Drawing" style="height: 150px"/>
# MAGIC 
# MAGIC When you click on the button, you will see the following popup.
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Publish1.png" alt="Drawing" />
# MAGIC 
# MAGIC When you click on "Publish", you will see a popup with your notebook's public link. **Copy the link and set the `notebook_URL` variable in the AUTOGRADER notebook (not this notebook).**
# MAGIC 
# MAGIC <img src="http://spark-mooc.github.io/web-assets/images/Lab0_Publish2.png" alt="Drawing" />

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2(d): Set the notebook URL and Lab ID in the Autograder notebook, and run it
# MAGIC 
# MAGIC Go to the Autograder notebook and paste the link you just copied into it, so that it is assigned to the `notebook_url` variable.
# MAGIC 
# MAGIC ```
# MAGIC notebook_url = "..." # put your URL here
# MAGIC ```
# MAGIC 
# MAGIC Then, find the line that looks like this:
# MAGIC 
# MAGIC ```
# MAGIC lab = <FILL IN>
# MAGIC ```
# MAGIC and change `<FILL IN>` to "CS120x-lab1a":
# MAGIC 
# MAGIC ```
# MAGIC lab = "CS120x-lab1a"
# MAGIC ```
# MAGIC 
# MAGIC Then, run the Autograder notebook to submit your lab.

# COMMAND ----------

# MAGIC %md
# MAGIC ### <img src="http://spark-mooc.github.io/web-assets/images/oops.png" style="height: 200px"/> If things go wrong
# MAGIC 
# MAGIC It's possible that your notebook looks fine to you, but fails in the autograder. (This can happen when you run cells out of order, as you're working on your notebook.) If that happens, just try again, starting at the top of Appendix A.
