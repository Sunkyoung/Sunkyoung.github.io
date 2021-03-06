---
emoji: ๐ฅ
title: Review AI504 Practice Session - 02 Basic ML
date: '2022-03-31 01:00:00'
author: ์ ๊ฒฝ
tags: Matplotlib LinearRegression Classification DeepLearning
categories: Deeplearning 
---

## 1. Matplotlib

NumPy ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ํ์ฉํ์ฌ ๊ทธ๋ํ๋ฅผ ๋ง๋ค์ด ์๊ฐํํ  ์ ์๋ ๋ผ์ด๋ธ๋ฌ๋ฆฌ์ด๋ค.

์ง๋๋ฒ NumPy์ ๋ํ ๊ธฐ๋ณธ์ ์ธ ์ค๋ช์ ์ฌ๊ธฐ์! ๐๐ผ

[https://sunkyoung.github.io/pytorch-study-01/](https://sunkyoung.github.io/pytorch-study-01/)

- How to use ?

```python
import matplotlib.pyplot as plt
import numpy as np
```

- Basic usage
    - plt.plot(x-axis, y-axis) : ์ฃผ์ด์ง x์ถ, y์ถ ๋ฐ์ดํฐ๋ฅผ ๋ฐํ์ผ๋ก ์ ํ ๊ทธ๋ํ๋ฅผ ๊ทธ๋ฆผ
    - plt.scatter(x-axis, y-axis, s=None, c=None) : ์ฃผ์ด์ง x์ถ, y์ถ ๋ฐ์ดํฐ๋ฅผ ๋ฐํ์ผ๋ก ์ ์ ๊ทธ๋ฆผ
        - s : ์ ์ ํฌ๊ธฐ
        - c : ์๊น ์ง์  (๋ฆฌ์คํธ ํํ๋ก๋ ์ง์ ๊ฐ๋ฅํ๋ฉฐ ๋ฆฌ์คํธ ๊ธธ์ด๋งํผ cmap๊ณผ norm์ mappingํ์ฌ ์์ ํํ)
        - e.g. `plt.scatter(X, y, s=30, c="red")`
    - plt.show() : ํ๋กฏ์ ๋ณด์ฌ์ค
    - np.linspace(start, end, number_of_sample) : sampling์ ์ํด ์ฃผ๋ก ์ฌ์ฉ๋๋ ํจ์์ด๋ฉฐ, start~end ๊ตฌ๊ฐ ๋ด์ ์๋ ๋ฐ์ดํฐ๋ค์ ์ง์ ํ ๊ฐ์(number_of_sample) ๋งํผ ๊ท ๋ฑํ๊ฒ ์ํ๋งํ์ฌ arrayํํ๋ก ๋ฐํํด์ค
        - end_point=False ๋ก ์ค์ ํ๋ค๋ฉด, ๋ฆฌ์คํธ์ ์ธ๋ฑ์ฑ๊ณผ ๊ฐ์ด start ~ (end-1) ๊ตฌ๊ฐ์ผ๋ก ์ค์ ๋๋ฉฐ, ๊ธฐ๋ณธ ๊ฐ์ end_point=True
        - e.g.
            
            ```python
            np.linspace(2.0, 3.0, num=5)
            # -> array([2.  , 2.25, 2.5 , 2.75, 3.  ])
            np.linspace(2.0, 3.0, num=5, endpoint=False)
            # -> array([2. ,  2.2,  2.4,  2.6,  2.8])
            ```
            
    

์์ ๊ธฐ๋ณธ ์ฌ์ฉ๋ฒ์ ๋ฐํ์ผ๋ก ๊ทธ๋ํ๋ฅผ ๊ทธ๋ ค๋ณด์!

```python
foo = lambda x: -(2/7*x**3-9/2*x**2+15*x-10.)

X = np.linspace(0, 10, 100)
y = foo(X)

x_sample = np.linspace(0, 10, 5)
y_sample = foo(x_sample)

plt.plot(X, y)
plt.scatter(x_sample, y_sample, c="red", s=30)
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.show()
```

โ ์ถ๋ ฅ๋๋ ๊ทธ๋ํ

![plot](img/plot.png)

## 2. Linear Regression

์ ํ ํ๊ท๋, ํ ๊ฐ ์ด์์ ๋๋ฆฝ ๋ณ์ X์ ์ข์ ๋ณ์ y ๊ฐ์ ์ ํ ์๊ด ๊ด๊ณ๋ฅผ ๋ชจ๋ธ๋งํ๋ ํ๊ท ๋ถ์ ๊ธฐ๋ฒ์ด๋ค. (์ถ์ฒ : [์ํคํผ๋์](https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%ED%9A%8C%EA%B7%80))

[Scikit-learn](https://scikit-learn.org/stable/index.html) ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ก ์ฝ๊ฒ ๋ง์ [์ ํ ํ๊ท](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)์ ๊ฐ์ ํต๊ณ์ ์ธ ๋ชจ๋ธ + ๊ธฐ๊ณํ์ต ๋ชจ๋ธ๋ค์ ์ ์ํ๊ณ  ์ฌ์ฉํ  ์ ์๋ค.

- Usage
    
    ```python
    from sklearn.linear_model import LinearRegression
    
    # ์ ํ ํ๊ท ๋ชจ๋ธ ์ ์
    lr = LinearRegression()
    
    foo = lambda x: -(2/7*x**3-9/2*x**2+15*x-10.)
    x_sample = np.linspace(0, 10, 5)
    y_sample = foo(x_sample)
    
    # ํ๋์ ๋ฐฐ์น ๋น ํ๋์ feature๋ฅผ ๊ฐ์ง๋๋ก ์ฐจ์ ์ถ๊ฐ
    x_new = x_sample[:, None]
    
    # ์ ํ ํ๊ท ๋ชจ๋ธ์ fittingํ์ฌ ํ์ต
    lr.fit(x_new, y_sample)
    
    # Coefficient ๊ณ์ฐ
    r2 = lr.score(x_new, y_sample)
    
    # y๊ฐ ์์ธก
    y_hat = lr.predict(x_new)
    # ๋ง์ฝ ํ๋์ ๋ฐ์ดํฐ ํฌ์ธํธ์ ๋ํ ์์ธก๊ฐ์ ์ป๊ณ  ์ถ๋ค๋ฉด
    # y_hat = lr.predict(x_new[0, None])
    
    # Mean Squared Error ๊ณ์ฐ
    MSE = np.mean((y_hat - y_sample)**2)
    
    plt.plot(x_new, y_hat)
    ```
    
    โ ์์ plot์ ๋ํด์ ๊ทธ๋ฆฐ ๊ฒฝ์ฐ์ ๋ํ ๊ทธ๋ํ
    
    ![plot+lr](img/plot+lr.png)
    

## 3. Polynomial Regression

๋คํญ ํ๊ท๋, 2์ฐจ ์ด์์ ๋คํญ์์ผ๋ก ์ด๋ฃจ์ด์ง ๋๋ฆฝ ๋ณ์ X์ ์ข์ ๋ณ์ y ๊ฐ์ ์๊ด ๊ด๊ณ๋ฅผ ๋ชจ๋ธ๋งํ๋ ํ๊ท ๋ถ์ ๊ธฐ๋ฒ์ด๋ค. (์ถ์ฒ : [์ํคํผ๋์](https://en.wikipedia.org/wiki/Polynomial_regression))

์์ ์ ํ ํ๊ท ์ค๋ช์์์ ๋์ผํ๊ฒ [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)์ผ๋ก ๊ตฌํ๋์ด ์์ด ์ฝ๊ฒ ์ฌ์ฉ ๊ฐ๋ฅํ๋ค.

- Usage

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_sample = np.linspace(0, 10, 5)
x_new = x_sample[:, None]

foo = lambda x: -(2/7*x**3-9/2*x**2+15*x-10.)
y_sample = foo(x_sample)

# 6์ฐจ ๋คํญ์์ผ๋ก ๋ณํ
poly = PolynomialFeatures(degree=6)
x_sample_poly = poly.fit_transform(x_new)
poly_lr = LinearRegression().fit(x_sample_poly, y_sample)

# -> x_new ์ถ๋ ฅ
# Before transform: (Single features)
# [[ 0. ]
#  [ 2.5]
#  [ 5. ]
#  [ 7.5]
#  [10. ]]

# -> x_sample_poly ์ถ๋ ฅ
# After transform: (Multiple features)
# [[1.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#   0.00000000e+00 0.00000000e+00 0.00000000e+00]
#  [1.00000000e+00 2.50000000e+00 6.25000000e+00 1.56250000e+01
#   3.90625000e+01 9.76562500e+01 2.44140625e+02]
#  [1.00000000e+00 5.00000000e+00 2.50000000e+01 1.25000000e+02
#   6.25000000e+02 3.12500000e+03 1.56250000e+04]
#  [1.00000000e+00 7.50000000e+00 5.62500000e+01 4.21875000e+02
#   3.16406250e+03 2.37304688e+04 1.77978516e+05]
#  [1.00000000e+00 1.00000000e+01 1.00000000e+02 1.00000000e+03
#   1.00000000e+04 1.00000000e+05 1.00000000e+06]]
```

์๋์ ๊ทธ๋ฆผ๊ณผ ๊ฐ์ด, ํญ์ ์ฐจ์๊ฐ ์ปค์ง ์๋ก sample๋ค์ ํน์ง์ ๋ ์ ๋ฐ์ํ์ฌ ๋ชจ๋ธ๋ง์ด ๊ฐ๋ฅํ์ง๋ง ๋๋ฌด ์ปค์ง๋ฉด Overfitting ๋๋ ๋ฌธ์ ๊ฐ ์๋ค. ๋ฐ๋๋ก, ํญ์ ์ฐจ์๊ฐ ๋๋ฌด ๋ฎ์ผ๋ฉด Underfitting ๋๋ ๋ฌธ์ ๊ฐ ์๋ค.

![polylr_overfitting_underfitting](img/polylr_overfitting_underfitting.png)

Overfitting์ด๋, training error๋ ์ ๊ณ  variance(์์ธก๊ฐ๋ค์ ํฉ์ด์ง ์ ๋)๊ฐ ํฐ ๋ฐ๋ฉด, test error๊ฐ ํฐ ํ์์ด ๋ํ๋ ๋ฌธ์ ๋ฅผ ๋งํ๋ค. ์ฆ, training data์ ๋ํด์๋ ์ข์ ์ฑ๋ฅ์ ๋ณด์ผ ์๋ ์์ด๋ ํ์ตํ์ง ์์ test data์ ๋ํด์๋ ์ข์ ์ฑ๋ฅ์ ๋ณด์ด์ง ๋ชปํ๋ค๋ ๋ฌธ์ ์ ์ ๊ฐ์ง๊ณ  ์๋ค. 

โ Overfitting ๋ฌธ์ ์ ๋ํ ๋ํ์ ์ธ ํด๊ฒฐ์ฑ์ผ๋ก๋ ๋ ๋ง์ ๋ฐ์ดํฐ๋ฅผ ์ฌ์ฉํ๊ฑฐ๋ Regularization(์ ๊ทํ)๋ฅผ ํ๋ ๋ฐฉ๋ฒ์ด ์๋ค.

Underfitting์ Overfitting๊ณผ๋ ๋ฐ๋๋ก training error์ test error๊ฐ ๋ ๋ค ํฌ๋ฉฐ, bias(์์ธก๊ฐ๊ณผ ์ ๋ต๊ฐ๊ณผ์ ์ฐจ์ด) ๋ํ ํฐ ํ์์ด ๋ํ๋ ๋ฌธ์ ๋ฅผ ๋งํ๋ค. ์ด๋ ์ถฉ๋ถํ ํ์ต๋์ง ์์ ๋ํ๋ ํ์์ด๋ค. 

โ Underfitting ๋ฌธ์ ์ ๋ํ ํด๊ฒฐํ๊ธฐ ์ํ ๋ํ์ ์ธ ๋ฐฉ๋ฒ์ผ๋ก๋ ๋ ๋ง์ feature๋ฅผ ์ถ๊ฐํ๊ฑฐ๋ ํญ์ ์ฐจ์๋ฅผ ์ฆ๊ฐํ์ฌ ๋ชจ๋ธ์ complexity๋ฅผ ๋์ด๊ฑฐ๋ ๋ ์ค๋ ํ์ตํ๋ ๋ฐฉ๋ฒ์ด ํด๊ฒฐ์ฑ์ด ๋  ์ ์๋ค.

### Regularization (์ ๊ทํ)

Overfitting ๋ฌธ์ ๋ฅผ ์ํํ๊ธฐ ์ํ ๋ฐฉ๋ฒ ์ค Regularization(์ ๊ทํ) ๋ฐฉ๋ฒ์ ๋ชจ๋ธ์ ์์ ๋๋ฅผ ์ ํํ์ฌ hypothesis space๋ฅผ ์ค์ธ๋ค. ์ฆ, ๋ชจ๋ธ์ ๊ฐ์ค์น๋ค์ ์ ํ์์ผ variance๋ฅผ ์ค์ด๊ณ , ์ผ๋ฐํํ๋ ๋ฅ๋ ฅ์ ๋์ธ๋ค. ๋ํ์ ์ธ ์ ๊ทํ ๋ฐฉ์์๋ 1) Ridge Regression 2) Lasso Regression 3) Elastic Net ์ธ ๊ฐ์ง๊ฐ ์๋ค. 

๊ฐ ๋ฐฉ์์ ๋ํด ํ๋์ฉ ์์๋ณด์ !

#### **Ridge Regression**

Ridge Regression (๋ฆฟ์ง ํ๊ท) ๋ฐฉ๋ฒ์ L2 Regression์ผ๋ก ๋ถ๋ฆฌ๋ฉฐ, ํ์ต ์ Cost function (๋น์ฉ ํจ์)์ Loss์ ๊ฐ์ค์น ๊ฐ๋ค์ ๋ํ L2 regularization term์ ๋ํ์ฌ ๋ชจ๋ธ์ ์์ ๋๋ฅผ ์ ํํ๋ค.

์๋ฅผ ๋ค์ด, 6์ฐจ์์ ๋คํญ์ ( $w^Tx+b = w_6x^6+w_5x^5+...+w_1x^1+ b$ ) ์ ๋ํด ๋คํญํ๊ท๋ฅผ ํ  ๋  

๋ฆฟ์ง ํ๊ท ๋ฐฉ๋ฒ์์์ ๋น์ฉ ํจ์๋ ์๋์ ๊ฐ์ด ํํ๋๋ค. 

$$
L(y,y')+\frac{\alpha}{2}\lVert w\rVert^2
$$

์ฌ๊ธฐ์์ $\alpha$ ๋ ์ ํ์ ์ ๋(penalty) ์ง์ ํ๋ hyperparameter์ด๋ค. 

$\alpha$ ๊ฐ 0์ด๋ฉด linear regression ์ด๊ณ , $\alpha$ ๊ฐ์ด ์ปค์ง ๊ฒฝ์ฐ ๋ชจ๋  ๊ฐ์ค์น๋ค์ด 0๊ณผ ๊ฐ๊น๊ฒ ๋์ด ์์ธก ๊ฐ์ด ๋ฐ์ดํฐ๋ค์ ํ๊ท ์ ๊ฐ๊น๊ฒ flat ํ ํํ๊ฐ ๋๋ค.

![ridge_alpha](img/ridge_alpha.png)

Scikit-learn ๊ตฌํ์
 ์๋์ ๊ฐ๋ค.

```python
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```

#### **Lasso Regression**

Lasso (**L**east **A**bsolute **S**hrinkage and **S**election **O**perator) Regression ๋ ์์ Ridge regression ์์ ์ฌ์ฉํ L2 regularization ๋์ , L1 regularization์ ํ๋ ๋ฐฉ๋ฒ์ด๋ค.

์์ ์ค๋ชํ 6์ฐจ์์ ๋คํญ์์ ํ ๋๋ก, ๋ผ์ ํ๊ท์์์ ๋น์ฉ ํจ์๋ ๋ค์๊ณผ ๊ฐ๋ค.

$$
L(y,y')+\alpha(|w_6|+|w_5|+...+|w_1|)
$$

๋ผ์ ํ๊ท์ ํฐ ํน์ง ์ค ํ๋๋ ์ค์ํ์ง ์์ feature๋ค์ ๊ฐ์ค์น๋ฅผ ์ ๊ฑฐํ๋ ๊ฒฝํฅ์ด ์๋ค๋ ๊ฒ์ด๋ค. 

- Why? ์ญ์ ํ(Backpropagation) ์์ ๋น์ฉ ํจ์์ ๋ํด ํธ๋ฏธ๋ถํ๊ฒ ๋๋ฉด, ๊ฐ์ค์น๋ ์์ ๊ฐ์ด ๋์ด๋ฒ๋ฆฌ๊ธฐ ๋๋ฌธ์, ๊ฐ์ค์น๊ฐ ๋๋ฌด ์์ ๊ฒฝ์ฐ 0์ ๊ฐ๊น์ ์ค์ํ์ง ์์ feature๋ค์ ๊ฐ์ค์น๋ค์ ์ ๊ฑฐํ๋ ํจ๊ณผ๋ฅผ ๋ํ๋ด๊ฒ ๋๋ค.

์์ ๋ฆฟ์ง ํ๊ท์ ๊ทธ๋ํ ์ด๋ฏธ์ง์ ์ ์ฌํ๋ฉด์๋, ๊ฐ์ค์น๋ฅผ ๋ ์๊ฒ ์ค์ ํ๋๋ผ๋ ๋ flatํ๊ฒ ๋ง๋๋ ๊ฒฝํฅ์ด ๋ํ๋๋ค. ์ฆ, ์๋์ ์ผ๋ก feature selectionํ๋ ํจ๊ณผ๋ฅผ ๋ํ๋ด๊ณ  sparse model์ ๋์ถํด๋ธ๋ค.

![lasso_alpha](img/lasso_alpha.png)

Scikit-learn์ผ๋ก ์ฝ๊ฒ ์ฌ์ฉ ๊ฐ๋ฅํ๋ค.

```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
```

#### **Elastic Net** 

Elastic Net์ Ridge Regression ๊ณผ Lasso Regression์ ์ค๊ฐ์ผ๋ก, ๋ regularization term์ ์์ด ์ฌ์ฉํ๋ค. mix ratio r ์ผ๋ก ์กฐ์ ํ  ์ ์๋ค.

$$
L(y,y')+r\alpha(\sum_{i=1}^n|w_i|)+\frac{l-r}{2}\alpha\lVert w\rVert^2
$$

์ธ ๊ฐ์ง ์ ๊ทํ ๋ฐฉ๋ฒ ์ค์ ๋ณดํต Ridge๋ฅผ ๊ธฐ๋ณธ์ผ๋ก ๋ง์ด ์ฌ์ฉ๋์ง๋ง, ์ผ๋ถ ์ ์ feature๋ง ์ ์ฉํ  ๋ Lasso ๋ Elastic Net์ ์ฌ์ฉํ๋ค. ๋ณดํต Lasso์ ๊ฒฝ์ฐ feature์ ๊ฐ์๊ฐ training instance๋ณด๋ค ๋ง๊ฑฐ๋ ์ผ๋ถ feature๋ค์ด correlate๋  ๋ ์์ธก์ด ์ด๋ ค์ฐ๋ฏ๋ก, Lasso ๋ณด๋ค Elastic Net์ด ์ ํธ๋๋ค.

Scikit-learn์ผ๋ก ์ฝ๊ฒ ์ฌ์ฉ ๊ฐ๋ฅํ๋ค.

```python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```

### ๋คํญ ํ๊ท vs ๋ฆฟ์ง ํ๊ท

๋คํญ ํ๊ท์ ๋ํด ๋ฆฟ์ง ์ ๊ทํ์ ์ ์ฉํ ๋ฆฟ์ง ํ๊ท ๋ํ [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ก ์ฝ๊ฒ ์ ์ฉ ๊ฐ๋ฅํ๋ค.

- Implementation
    
    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    
    x_line = np.linspace(0, 10, 100)
    x_sample = np.linspace(0, 10, 5)
    x_new = x_sample[:, None]
    
    foo = lambda x: -(2/7*x**3-9/2*x**2+15*x-10.)
    Y = foo(x_line)
    y_sample = foo(x_sample)
    # 6์ฐจ ๋คํญ์์ผ๋ก ๋ณํ
    poly = PolynomialFeatures(degree=6)
    x_sample_poly = poly.fit_transform(x_new)
    x_line_poly = poly.fit_transform(x_line[:, None])
    
    # ๋คํญ ํ๊ท ์ ์ฉ ๋ฐ ์์ธก
    poly_lr = LinearRegression().fit(x_sample_poly, y_sample)
    y_poly = poly_lr.predict(x_line_poly)
    
    # Ridge ํ๊ท ์ ์ฉ ๋ฐ ์์ธก
    # penalty์ ์ ๋๋ฅผ ๋ํ๋ด๋ lambda๋ 'alpha'๋ผ๋ hyperparameter๋ก ์ฌ์ฉ๋จ !
    rr = Ridge(alpha=10.0).fit(x_sample_poly, y_sample)
    y_poly_rr = rr.predict(x_line_poly)
    ```
    
    โ Plot์ผ๋ก ํํ
    
    (blue: ์ ๋ต ๊ฐ, orange: ๋คํญ ํ๊ท ์์ธก ๊ฐ, green: ๋ฆฟ์ง ํ๊ท ๊ฐ)
    
    ![plot_poly_ridge](img/plot_poly_ridge.png)
    

## 4. Classification (Logistic Regression, Support Vector Machine, Decision Tree)

๋ํ์ ์ธ Classification(๋ถ๋ฅ) ๋ชจ๋ธ์ผ๋ก๋ ๋ค์๊ณผ ๊ฐ์ด ์ธ ๊ฐ์ง๊ฐ ์๋ค.

- Logistic Regression (๋ก์ง์คํฑ ํ๊ท)
- Support Vector Machine (SVM) (์ํฌํธ ๋ฒกํฐ ๋จธ์ )
- Decision Tree (๊ฒฐ์  ํธ๋ฆฌ)

### Logistic Regression (๋ก์ง์คํฑ ํ๊ท)

๋ก์ง์คํฑ ํ๊ท๋ ์ ํ ํ๊ท์ ๊ฐ์ด ์๋ ฅ feature๋ค์ weighted sum์ ๊ณ์ฐํ์ง๋ง, ๊ฒฐ๊ณผ๋ฅผ ๋ฐ๋ก ์ถ๋ ฅํ๋ ๊ฒ์ด ์๋๋ผ ๊ฒฐ๊ณผ์ logistic์ ์ถ๋ ฅํ๋ค. logistic ์ sigmoid function์ ๋ปํ๋ฉฐ 0๊ณผ 1 ์ฌ์ด์ ๊ฐ์ ๊ฐ์ง๋ค. 

![sigmoid](img/sigmoid.png)

- logit (log-odds) : logistic function์ ์ญ์ผ๋ก, positive class์ธ์ง negative class์ธ์ง ์ธก์ ๋ ํ๋ฅ (p)์ ratio์ log ๊ฐ์ด๋ค.
    
    $$logit(p) = log(p/(1-p))$$

closed-form equation์ด ์๋๊ธฐ ๋๋ฌธ์, ๋น์ฉํจ์๋ฅผ ์ต์ํ ํ๋ parameter๋ฅผ ๊ณ์ฐํ๋ ๊ฒ์ด ์ด๋ ต์ง๋ง, ๋น์ฉํจ์๊ฐ convex ํํ์ผ ๊ฒฝ์ฐ Gradient Descent์ ๊ฐ์ ์ต์ ํ ์๊ณ ๋ฆฌ์ฆ์ ์ฌ์ฉํ๋ค๋ฉด global optimum์ ์ฐพ์ ์ ์๋ค.

- multiple classes์ ๊ฒฝ์ฐ, softmax regression(multinomial logistic regression)์ผ๋ก ์ฌ์ฉํ  ์ ์๋ค. ์ด ๋, ํด๋์ค์ ํด๋์ค ๋ณ ํ๋ฅ ์ matchํ์ฌ ์ธก์ ํ๋ cross entropy ๋น์ฉ ํจ์๋ฅผ ์ฌ์ฉํ๋ค.

### Support Vector Machine (SVM) (์ํฌํธ ๋ฒกํฐ ๋จธ์ )

์ํฌํธ ๋ฒกํฐ ๋จธ์  ๋ถ๋ฅ๋ ์ฝ๊ฒ ์ค๋ชํ์๋ฉด, ๋ถ๋ฅํ๊ณ ์ ํ๋ ๋ ํด๋์ค ๊ฐ์ ๊ฑฐ๋ฆฌ(margin)๋ฅผ ์ต๋ํํ๋ ๋ชจ๋ธ์ด๋ค. small or medium ํฌ๊ธฐ์ ๋ฐ์ดํฐ ์์ ์ ํฉํ ๋ชจ๋ธ์ด๋ค

![svm](img/svm.png)

๋ ์นดํ๊ณ ๋ฆฌ ์ค ์ด๋ ํ๋์ ์ํ ๋ฐ์ดํฐ์ ์งํฉ์ด ์ฃผ์ด์ก์๋, ์ฃผ์ด์ง ๋ฐ์ดํฐ ์งํฉ์ ๋ฐํ์ผ๋ก ์๋ก์ด ๋ฐ์ดํฐ๊ฐ ์ด๋ ์นดํ๊ณ ๋ฆฌ์ ์ํ ์ง ํ๋จํ๋ ๋นํ๋ฅ ์  ์ด์ง ์ ํ ๋ถ๋ฅ ๋ชจ๋ธ์ ๋ง๋ ๋ค. ์ด ๋, ๋ฐ์ดํฐ๊ฐ ์๋ฒ ๋ฉ๋ ๊ณต๊ฐ์์ ๊ฒฝ๊ณ(boundary)๋ฅผ ํํํ  ๋ ๊ฐ์ฅ ํฐ ํญ(large margin)์ ๊ฐ์ง ๊ฒฝ๊ณ๋ฅผ ์ฐพ๋๋ค. ์ฆ, ๊ฐ์ฅ ๊ฐ๊น์ด ๊ฐ ํด๋์ค์ ๋ฐ์ดํฐ ์ ๋ค ๊ฐ์ ๊ฑฐ๋ฆฌ๋ฅผ ์ต๋๋ก ํ๋ค.

๋๋ฌธ์, ํ์ต์ด ์งํ๋๋ ๋์ SVM์ ๊ฐ ํ๋ จ ๋ฐ์ดํฐ ํฌ์ธํธ๊ฐ ๋ ํด๋์ค ์ฌ์ด์ ๊ฒฐ์  ๊ฒฝ๊ณ๋ฅผ ๊ตฌ๋ถํ๋ ๋ฐ ์ผ๋ง๋ ์ค์ํ ์ง๋ฅผ ๋ฐฐ์ฐ๊ฒ ๋๋ค. ๋ฐ์ดํฐ์ ์ ์ฒด๊ฐ ์๋ ํด๋์ค ์ฌ์ด์ ๊ฒฝ๊ณ์ ์์นํ ๋ฐ์ดํฐ ํฌ์ธํธ๋ค์ด ๊ฒฐ์  ๊ฒฝ๊ณ๋ฅผ ๋ง๋๋ ๋ฐ ์ํฅ์ ์ค๋ค. ์ด๋ฌํ ๋ฐ์ดํฐ ํฌ์ธํธ๋ค์ ์ํฌํธ ๋ฒกํฐ(support vector)๋ผ๊ณ  ํ๋ค.

์ ํ ๋ถ๋ฅ ๋ฟ๋ง ์๋๋ผ ๋น์ ํ ๋ถ๋ฅ์์๋ ์ฌ์ฉ๋  ์ ์์ผ๋ฉฐ, ์ ํ์ ์ผ๋ก ๋ถ๋ฅ๊ฐ ์ด๋ ค์ด ๋ฐ์ดํฐ์ Feature๋ฅผ ๋ํด(polynomial feature) ๊ณ ์ฐจ์ ๊ณต๊ฐ์ผ๋ก ๋์์์ผ ๋ถ๋ฆฌ๋ฅผ ์ฝ๊ฒ ํ๋ ๋ฐฉ๋ฒ์ ์ฌ์ฉํ๋ค. polynomial degree๊ฐ ํด ์๋ก ๋ชจ๋ธ์ด ๋๋ ค์ง๊ธฐ ๋๋ฌธ์, ๋ฌธ์ ์ ์ ์ ํ kernel trick์ ์ฌ์ฉํ๋ค. ์ด๋ ์ ๋ค์ ์งํฉ๊ณผ ์์ ๋ฒกํฐ์ ๋ด์  ์ฐ์ฐ์ผ๋ก ์ ์ํ์ฌ ํจ์จ์ ์ผ๋ก ๊ณ์ฐํ๋๋ก ๋๋๋ค. ํฐ ๋ฐ์ดํฐ์์ ๊ฒฝ์ฐ, Gaussian RBF Kernel์ ์ฌ์ฉํ๋ค.

![svm_kernel](img/svm_kernel.png)

์ฅ์  : ๋ถ๋ฅ, ์์ธก์ ์ฌ์ฉ ๊ฐ๋ฅ. overfitting ์ ๋๊ฐ ๋ํ๋ค. ์์ธก์ ์ ํ๋๊ฐ ๋๊ณ , ์ฌ์ฉํ๊ธฐ ์ฌ์

๋จ์  : kernel, parameter ์กฐ์  ํ์คํธ๋ฅผ ์ฌ๋ฌ๋ฒ ํด์ผ ์ต์ ํ๋ ๋ชจ๋ธ์ ๋ง๋ค ์ ์์, ๋ชจ๋ธ ๊ตฌ์ถ ์๊ฐ ์ค๋๊ฑธ๋ฆผ

### Decision Tree (๊ฒฐ์  ํธ๋ฆฌ)

๊ฒฐ์  ํธ๋ฆฌ ์๊ณ ๋ฆฌ์ฆ์ Feature์ ๋ํด ์๋์ ๊ทธ๋ฆผ๊ณผ ๊ฐ์ด ํธ๋ฆฌ ์๋ฃ ๊ตฌ์กฐ ๊ธฐ๋ฐ์ผ๋ก ๋ถ๋ฅํ๋ค. 

![dt](img/dt.png)

๊ฒฐ์  ํธ๋ฆฌ๋ feature scaling์ด๋ centering๊ณผ ๊ฐ์ ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ๊ฐ ํ์ํ์ง ์๋ค.

Scikit-Learn ๋ผ์ด๋ธ๋ฌ๋ฆฌ์์์ ๊ตฌํ์ Classification and Regression Tree(CART) ์๊ณ ๋ฆฌ์ฆ์ ๊ธฐ๋ฐ์ผ๋ก ํ์ตํ๋ค.

๋จผ์ , ํ์ต ๋ฐ์ดํฐ์ ๋ํด ํ๋์ feature $k$ ์ ๊ทธ์ ๋ํ threshold $t_k$ ๋ฅผ ๊ธฐ์ค์ผ๋ก ๋ ๊ฐ์ subset์ผ๋ก ๋๋๋ค. ์ด ๋, threshold๋ ์ ๋ถ๋ฆฌ๋(Purest) Subset์ด ๋๋๋ก $(k, t_k)$ ์์ ์ฐพ์์ ์ค์ ํ๋ค.

$$
J(k,t_k)=\frac{m_{left}}{m}G_{left}+\frac{m_{right}}{m}G_{right}
$$


$G_{left/right}$ ๋ left, right subset์ ์ ๋ถ๋ฆฌ๋์ง ์์์ ์ ๋(impurity)๋ฅผ ๋ปํ๊ณ , $m_{left/right}$ ์ left, right ๊ฐ subset์ ๊ฐ์๋ฅผ ๋ปํ๋ค. 

Regression ๋ฌธ์ ์ ์ ์ฉํ๋ค๋ฉด, $G_{left/right}$  ๋์  $$MSE_{left/right}$ loss๋ฅผ ์ฌ์ฉํ๋ค.

$$
MSE_{node} = \sum_{i \in node}(\hat y_{node} - y^{(i)})^2 \\ \hat y_{node}= \frac{1}{m_{node}}\sum_{i \in node}y^{(i)}
$$

**max_depth** parameter๋ก ๋์ด, ์์ ๊ณผ์ ์ ์ฌ๊ท์ ์ผ๋ก ๋ฐ๋ณตํ๋ฉฐ depth๋งํผ์ ํธ๋ฆฌ๋ฅผ ๊ตฌ์ถํ๋ค.

์ ํ ๋ชจ๋ธ๊ณผ ๋ฌ๋ฆฌ ๊ฒฐ์  ํธ๋ฆฌ ๋ชจ๋ธ์ ๊ฒฝ์ฐ ํ์ต ๋ฐ์ดํฐ์ ๋ํ ์ ํํ๋ ์ ๋๊ฐ ์ ๊ธฐ ๋๋ฌธ์, ํ์ต ๋ฐ์ดํฐ์ ํธ๋ฆฌ ๊ตฌ์กฐ๊ฐ ๋๋ฌด adaptํ๊ฒ ํ์ต ๋  ๊ฒฝ์ฐ์ overfitting ๋ฌธ์ ๊ฐ ๋ฐ์ํ  ์ ์๋ค. ํ๋ผ๋ฏธํฐ๊ฐ ์๋ค๊ธฐ ๋ณด๋ค ํ๋ผ๋ฏธํฐ๋ก ์ธํ ์ ํ์ด ํ์ต ์ด์ ์ ์๋ ๋ชจ๋ธ์ nonparametric ๋ชจ๋ธ์ด๋ผ๊ณ  ๋ถ๋ฅธ๋ค. ๋ฐ๋๋ก parametric ๋ชจ๋ธ์ degree of freedom์ด ์ ํ๋๊ธฐ ๋๋ฌธ์ overfitting์ ์ํ์ ์ค์ผ ์ ์๋ค. overfitting์ ์ค์ด๊ธฐ ์ํด์ freedom์ ์ ํํ๊ธฐ ์ํด ์ ๊ทํ(regularization)์ ์ ์ฉํ๋ค. ์์ธํ hyperparameter๋ [์ฌ๊ธฐ](https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use)์ ์ค๋ช๋์ด ์๋ค.

### Implementation

์์ ์ธ ๋ชจ๋ธ ([Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html#), [Decision Tree](https://scikit-learn.org/stable/modules/tree.html#)) ๋ชจ๋ Scikit-learn ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ก ๊ฐ๋จํ๊ฒ ๊ตฌํํ์ฌ ์ฌ์ฉํ  ์ ์๋ค.

```python
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(random_state=1234)
logistic.fit(X_train[:, :2], y_train)
# softmax regression
# softmax_reg = LogisticRegression(multi_class="multinomial")

from sklearn.svm import SVC

svm = SVC(kernel='linear', random_state=1234)
svm.fit(X_train[:, :2], y_train)

from sklearn.tree import DecisionTreeClassifier
# more depth, increase decision boundary - leads to overfitting 
# similar to polynomial
dt = DecisionTreeClassifier(max_depth=2, random_state=1234) 
dt.fit(X_train[:, :2], y_train)
```

๐๐ผ ๊ด๋ จ ์ค์ต ์ฝ๋ :

[https://github.com/Sunkyoung/PyTorch-Study/blob/main/PyTorch_Study_02_Basic_ML.ipynb](https://github.com/Sunkyoung/PyTorch-Study/blob/main/PyTorch_Study_02_Basic_ML.ipynb)

**Reference**

Aurelien Geron, Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, Oโreilly (2019)

```toc

```