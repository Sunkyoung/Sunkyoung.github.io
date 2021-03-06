---
emoji: ð¥
title: Review AI504 Practice Session - 01 NumPy
date: '2022-01-10 09:30:00'
author: ì ê²½
tags: NumPy DeepLearning
categories: Deeplearning 
---

## 0. NumPy
íë ¬ì´ë ì¼ë°ì ì¼ë¡ ëê·ëª¨ ë¤ì°¨ìÂ ë°°ì´ì ì½ê² ì²ë¦¬ í  ì ìëë¡ ì§ìíëÂ íì´ì¬ìÂ ë¼ì´ë¸ë¬ë¦¬ì´ë¤. (ì¶ì²: [ìí¤í¼ëì](https://ko.wikipedia.org/wiki/NumPy))
- How to use? `import numpy as np`

## 1. NumPy array data

- Scalar : single number
    - e.g. `a = np.array(1.)`
- Vectors : an array of numbers
    - e.g. `b = np.array([1., 2., 3.])`
- Matrix : 2-D array
    - e.g. `c = np.array([[1., 2., 3.], [4., 5., 6.]])`
- Tensor : N-dimensional array (n â¥ 2)
    - e.g. `d = np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]])`

![[https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.1-Scalars-Vectors-Matrices-and-Tensors/](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.1-Scalars-Vectors-Matrices-and-Tensors/)](./scalar,vector,matrix,tensor.png)
*[https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.1-Scalars-Vectors-Matrices-and-Tensors/](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.1-Scalars-Vectors-Matrices-and-Tensors/)*

### Functions

- `.ndim` : show dimension
    
    e.g. 
    
    `a.ndim` â 0 
    
    `b.ndim` â 1
    
    `c.ndim` â 2
    
    `d.ndim` â 3
    
- `.shape` : show the number of values for each dimension
    
    e.g.
    
    `a.shape` â ()
    
    `b.shape` â(3,)
    
    `c.shape` â (2,3)
    
    `d.shape` â (2, 2, 3)
    

>ð¡ **NOTE :** â ë¤ë **ì¶ë ¥ê°(Output)**ì ìë¯¸



## 2. Define Numpy arrays

- np.ones(*shape*) : define array given shape and fill with 1
    
    e.g. (10,) shapeë¥¼ ê°ì§ë©´ì 1ì¼ë¡ ì±ìì§ ë°°ì´ì ì ì
    
    usage :  `np.ones(10)`
    
- np.zeros(*shape*) : define array given shape and fill with 0
    
    e.g. (2,5) shapeë¥¼ ê°ì§ë©´ì 0ì¼ë¡ ì±ìì§ ë°°ì´ì ì ì
    
    usage : `np.zeros((2,5))`
    
- np.full(*shape, number*) :  define array given shape and fill with given number
    
    e.g. (2,5) shapeë¥¼ ê°ì§ë©´ì 5ë¡ ì±ìì§ ë°°ì´ì ì ì 
    
    usage : `np.full((2,5),5)`
    
- np.random.random(*shape*) : define array given shape and fill with random numbers
    
    e.g. (2,3,4) shapeë¥¼ ê°ì§ë ëë¤ ë°°ì´ ì ì 
    
    usage :  `np.random.random((2, 3, 4))`
    
- np.arange(*number*) : define array which contains 0 to given number-1 (similar to python `range()`)
    
    e.g. 0~9ë¡ êµ¬ì±ë ë°°ì´
    
    usage : `np.arrange(10)`
    
    â array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    - np.arange(*number*).astype(*type*) : define arange() array given data type
        
        e.g. float íìì 0~9ë¡ êµ¬ì±ë ë°°ì´
        
        usage : `np.arange(10).astype(float)`
        
        â array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        
    - np.arrange(*number*).reshape(*shape*) : define arange() array given shape
        
        e.g.(5, 2) shapeë¥¼ ê°ì§ë©´ì 0~9ë¡ êµ¬ì±ë ë°°ì´
        
        usage : `np.arange(10).reshape((5,2))`
        
        â array([[0, 1],
                        [2, 3],
                        [4, 5],
                        [6, 7],
                        [8, 9]])
        


>ð¡ **NOTE :** shapeë¥¼ ëªìí  ë, shapeê° 2-by-3 ì´ë©´, ê´í¸ê¹ì§ í¬í¨í´ì **(2,3)**ë¡ ëªìíê¸°! <br> &emsp; &emsp; &emsp; &emsp; e.g. ðð»ââï¸Â **np.ones(2,3)**  ðð»ââï¸Â **np.ones((2,3))**



## 3. Indexing & Slicing

pythonììì indexing & slicing ê³¼ ëì¼íê² ìëíë¤.

e.g. `a = np.arange(10)` `b = np.arange(9).reshape(3,3)`

- a[index] : access index of a, and index could be < 0
    
    e.g. aì 0ë²ì§¸ ì¸ë±ì¤ì ìë ê°
    
    usage : `a[0]` â 0
    
    e.g. aì ë¤ìì 4ë²ì§¸ì ìë ê°
    
    usage : `a[-4]` â 6
    
    e.g. ë§ì§ë§ row
    
    usage : `b[-1]` â [6 7 8]
    
    - Conditional indexing : ì¡°ê±´ë¬¸ì ë°ë¥¸ ì¸ë±ì± ê°ë¥
        
        e.g. ì§ìì¸ ê°ë§ ì¶ë ¥
        
        ```python
        idx = b % 2 == 0
        # -> [[True False  True]
        #     [False  True False]
        #     [True False  True]]
        b[idx]
        # -> [0 2 4 6 8]
        ```
        
    - Specific elements from a nd-array
        
        e.g. [0, 2, 3] indexì ëí ê° ì¶ë ¥ (vector)
        
        ```python
        idx = [0, 2, 3]
        a[idx]
        # [0 2 3]
        ```
        
        e.g. [0, 2] í í¹ì ì´ì í´ë¹íë ê° ì¶ë ¥ (tensor)
        
        ```python
        idx = [0, 2]
        # row
        b[idx, :]
        # -> [[0 1 2]
        #     [6 7 8]]
        
        # column
        b[:,idx]
        # -> [[0 2]
        #     [3 5]
        #     [6 8]]
        ```
        
        e.g. [[0,0,1],[1,2,0]]ì í´ë¹íë ê° ì¶ë ¥ (tuple ííë ê°ë¥)
        
        ```python
        idx = np.array([[0,0,1],[1,2,0]])
        # tuple : idx = ((0,0,1),(1,2,0))
        b[idx]
        # -> [[[0 1 2]
        #      [0 1 2]
        #      [3 4 5]]
        #     [[3 4 5]
        #      [6 7 8]
        #      [0 1 2]]]
        ```
        
- a[*start index*:*end index*:*interval*] : access from start index to end index - 1 with interval, and interval could be < 0
    
    e.g. 2ë²ì§¸ë¶í° 4ë²ì§¸ ì¸ë±ì¤ì ìë ê°
    
    usage : `a[2:5]` â [2 3 4]
    
    e.g. 0ë²ì§¸ë¶í° 10ë²ì§¸ ì¸ë±ì¤ ì´ì ê¹ì§ ìë ê°ì 3ì ê°ê²©ì¼ë¡ ì¶ë ¥
    
    usage : `a[0:10:3]` â [0 3 6 9]
    
    e.g.  8ë²ì§¸ë¶í° 5ë²ì§¸ ì¸ë±ì¤ ë¤ìê¹ì§ ìë ê°ì -1ì ê°ê²©ì¼ë¡ ì¶ë ¥
    
    usage : `a[8:5:-1]` â [8 7 6]
    
    e.g. ëë²ì§¸ column
    
    usage : `b[:,1]` â [1 4 7]
    

## 4. Math Operations

- Element-wise operation
    - +, - , x, / : given two numpy arrays should have same shape
        
        e.g. 
        
        ```python
        a = np.arange(6).reshape((3, 2))
        b = np.full((3,2),2)
        
        print(a+b)
        # -> [[2 3]
        #     [4 5]
        #     [6 7]]
        
        print(a-b)
        # -> [[-2 -1]
        #     [ 0  1]
        #     [ 2  3]]
        
        print(a*b)
        # -> [[ 0  2]
        #     [ 4  6]
        #     [ 8 10]]
        
        print(a/b)
        # -> [[0.  0.5]
        #     [1.  1.5]
        #     [2.  2.5]]
        ```
        
- Unary operation
    
    e.g. `a = np.arange(6).reshape((3,2))`
    
    - a.sum(axis) : get sum of a regarding axis (axis is optional)
        
        usage : `a.sum()` â 15
        
        e.g. axis 0ì ëí sum (row-wise)
        
        usage : `a.sum(axis=0)` â [6 9]
        
        e.g. axis 1ì ëí sum (column-wise)
        
        usage : `a.sum(axis=1)` â [1 5 9]
        
    - `a.mean()` : get mean of a â 2.5
    - `a.max()` : get maximum value of a â 5
    - `a.min()` : get minimum value of a â 0
- Dot product, Multiplication
    
    e.g. 
    
    vector : `a = np.arange(3).astype('float')` `b = np.ones(3)`
    
    matrix : `a = np.arange(6).reshape((3, 2))` `b = np.arange(6).reshape((2, 3))`
    
    tensor : `a = np.arange(24).reshape((4, 3, 2))` `b = np.ones((4, 2, 3))`
    
    - Dot product
        
        usage : `np.dot(a, b)`
        
        - vector : `np.dot(a, b)` â 3.0
        - matrix : `np.dot(a, b).shape` â (3, 3)
        - tensor : `np.dot(a, b).shape` â (4, 3, 4, 3)
    - Mutiplication
        
        usage : `a@b`
        
        - vector : `a@b` â 3.0
        - matrix : `(a@b).shape` â (3, 3)
        - tensor : `(a@b).shape` â (4, 3, 3)

## 5. Shape Manipulation

- Reshape
    
    usage : `a.reshape(shape)`
    
    - shape ìì -1 ì ì£¼ì´ì§ ìê° ìì ê²½ì°, íì¬ shapeìì ì£¼ì´ì§ ìë§í¼ì ëª¨ìì ê°ì§ê³  ë¨ë ì°¨ìì¼ë¡ í ë¹íê³ , ì£¼ì´ì§ ìê° ìì ê²½ì°, 1ì°¨ì ë°°ì´ë¡ ë§ë ë¤. ê·¸ë¦¬ê³  -1ì íµí´ ê°ë¥í ì°¨ì ì¶ê°ë í ì°¨ì ì¶ê°ë§ ê°ë¥íë¤.
    
    e.g. 
    
    ```python
    a = np.arange(24)
    b = a.reshape((6, 4))
    print(b.shape)
    c = a.reshape((6, -1))
    print(c.shape)
    # bì cì shapeë (6,4)ë¡ ëì¼
    d = a.reshape((6,4,-1))
    print(d.shape)
    # dì shapeë (6, 4, 1)ì´ ë¨
    # d = a.reshape((6,4,-1,-1)) -> ValueError: can only specify one unknown dimension
    e = b.reshape(-1)
    # eì shapeë aììì shapeì ëì¼íê² (24,)ê° ë¨
    ```
    
- Add an extra dimension
    
    usage : `a[:, None]`
    
    - ì°¨ì ì¶ê°íê¸° ìíë ë¶ë¶ì Noneì ìë ¥íê³ , ê¸°ì¡´ ê°ë¤ì [:] slicingì íµí´ ë³µì¬
    
    e.g.
    
    ```python
    a = np.arange(3)
    print(a.shape) # -> (3,)
    b = a[:, None]
    print(b)
    # -> [[0]
    #     [1]
    #     [2]]
    print(b.shape) # -> (3, 1)
    c = a[None, :]
    print(c) # -> [0 1 2]
    print(c.shape) # -> (1, 3)
    d = a[:, None, None]
    print(d.shape) # -> (3, 1, 1)
    ```
    
- Stack, concatenation
    - vstack : stack vertically
        
        usage : `np.vstack(tuple)`
        
    - hstack : stack horizontally
        
        usage : `np.vstack(tuple)`
        
    - concatenate : concatenate on axis (default axis = 0)
        
        usage : `np.concatenate(tuple, axis)`
        
        - axis = 0 ì´ë©´ vstackì ê²°ê³¼ì ëì¼íê³ , axis = 1 ì´ë©´ hstackì ê²°ê³¼ì ëì¼
        - axis = None ì´ë©´ 1ì°¨ìì¼ë¡ ë§ë¤ì´ë²ë¦¼
        - axis ì í´ë¹íë dimensionì ì ì¸íê³  ë¬´ì¡°ê±´ ëë¨¸ì§ input dimensionì ëì¼í´ì¼ í¨!
    
    e.g.
    
    ```python
    a = np.ones((3,2))
    b = np.zeros((3,2))
    
    print(np.vstack([a, b]))
    # -> [[1. 1.]
    #     [1. 1.]
    #     [1. 1.]
    #     [0. 0.]
    #     [0. 0.]
    #     [0. 0.]]
    print(np.hstack([a, b]))
    # -> [[1. 1. 0. 0.]
    #     [1. 1. 0. 0.]
    #     [1. 1. 0. 0.]]
    print(np.hstack([a, b, a]))
    # -> [[1. 1. 0. 0. 1. 1.]
    #     [1. 1. 0. 0. 1. 1.]
    #     [1. 1. 0. 0. 1. 1.]]
    print(np.concatenate([a, b], axis=0))
    # -> [[1. 1.]
    #     [1. 1.]
    #     [1. 1.]
    #     [0. 0.]
    #     [0. 0.]
    #     [0. 0.]]
    print(np.concatenate([a, b], axis=1))
    # -> [[1. 1. 0. 0.]
    #     [1. 1. 0. 0.]
    #     [1. 1. 0. 0.]]
    
    a = np.ones((4,3,2))
    b = np.zeros((5,4,2))
    np.concatenate([a,b], axis=2) # -> ValueError
    
    a = np.ones((4,3,2))
    b = np.zeros((4,3,7))
    np.concatenate([a,b], axis=2).shape # -> (4,3,9)
    ```
    

- Transpose
    - Matrix transpose
        
        usage : `a.T`
        
        e.g.
        
        ```python
        a = np.arange(6).reshape((3, 2))
        b = a.T
        print(b.shape) # -> (2, 3)
        ```
        
    - Tensor transpose
        
        usage : `np.transpose(a, axes)`
        
        - axes : ì ì²´ axis ê°¯ìì tuple or list of ints ë¡, ìíë ììì axis ë¤ì ëì´
        
        e.g.
        
        ```python
        a = np.arange(24).reshape((4, 3, 2))
        b = np.transpose(a, [0, 2, 1])
        print(b.shape) # -> (4, 2, 3)
        c = np.transpose(a, [1, 0, 2])
        print(c.shape) # -> (3, 4, 2)
        ```
        

## 6. Broadcasting

Broadcasting : ë ë°°ì´ì´ ë¤ë¥¸ shapeë¥¼ ê°ì§ëë¼ë ë ë°°ì´ì shapeê° ì¡°ê±´ì ë§ì¡±íë¤ë©´, ë ì¤ ë ìì dimensionì´ ë í° dimensionì ê°ì§ëë¡ ë§ë¤ì´ì ì¬ì¹ì°ì°ì´ ê°ë¥íëë¡ íë ê²ì ë§íë¤.

- ì¡°ê±´ : ëì¼í dimensionì´ê±°ë 1ì´ì´ì¼ íë¤.

![[https://numpy.org/doc/stable/user/basics.broadcasting.html](https://numpy.org/doc/stable/user/basics.broadcasting.html)](./broadcasting_example.png)
*[https://numpy.org/doc/stable/user/basics.broadcasting.html](https://numpy.org/doc/stable/user/basics.broadcasting.html)*

e.g.

```python
# Vector and scalar
a = np.arange(3)
b = 2.
print(a+b) # -> [2. 3. 4.]
print(a-b) # -> [-2. -1.  0.]
print(a*b) # -> [0. 2. 4.]
print(a/b) # -> [0.  0.5 1. ]

# Matrix and vector
a = np.arange(6).reshape((3, 2))
b = np.arange(2).reshape((1, 2))
print((a+b).shape) # -> (3, 2)

# Tensor and matrix
a = np.arange(12).reshape((2,3,2))
b = np.arange(6).reshape((3,2))
print((a+b).shape) # -> (2, 3, 2)
print(a + b[None,:]) # -> (2, 3, 2)
```

## Quiz : Fill the function `foo()`

```python
import numpy as np

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def foo(M, W):
	# Define a function that, given M of shape (m,n) and W of shape (4n, n), executes the following:
	# - Take the first half rows of M
	# - Take the second half rows of M
	# - Take the odd-numbered rows of M
	# - Take the even-numbered rows of M
	# - Append them horizontally in the listed order so that you obtain a matrix X of shape (?, 4n)
	# - Linearly transform X with W so that you obtain a matrix Y of shape (?, ?)
	# - Put Y through the sigmoid function
	# - Obtain the sum of the row-wise mean
	
W = np.arange(16).reshape(8,2).astype('float') / 10.
M = (np.arange(20).reshape((10,2)).astype('float') - 10.) / 10.

foo(M, W)
```

<details>
<summary>Answer</summary>
<div markdown="1">

 ```python
    import numpy as np
    
    def sigmoid(x):
        return 1./(1. + np.exp(-x))
    
    def foo(M, W):
      (row, col) = M.shape
      first_half_rows = M[:row//2]
      second_half_rows = M[row//2:]
      odd_num_rows = M[1::2]
      even_num_rows = M[0::2]
      X = np.hstack([first_half_rows, second_half_rows, odd_num_rows, even_num_rows])
      # X = np.concatenate([first_half_rows, second_half_rows, odd_num_rows, even_num_rows], axis=1)
      Y = np.dot(X, W)
      sigmoid_Y = sigmoid(Y)
      row_wise_mean = sigmoid_Y.mean(axis=0)
      return np.sum(row_wise_mean)
    
    W = np.arange(16).reshape(8,2).astype('float') / 10.
    M = (np.arange(20).reshape((10,2)).astype('float') - 10.) / 10.
    
    foo(M, W)
```
</div>
</details>

<br>

ðð¼  ê´ë ¨ ì¤ìµ ì½ë : <https://github.com/Sunkyoung/PyTorch-Study/blob/main/PyTorch_Study_01_NumPy.ipynb>

```toc

```