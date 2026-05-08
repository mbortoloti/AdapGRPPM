

# An adaptative proximal point method for nonsmooth and nonconvex optimization on Hadamard manifolds

## Vitaliano Amaral, Marcio Bortoloti, Jurandir O. Lopes, Gilson Silva


**Abstract** This paper addresses a class of nonsmooth and nonconvex optimization problems defined on complete Riemannian manifolds. The objective function has a composite structure, combining convex, differentiable, and lower semicontinuous terms, thereby generalizing the classical framework of difference-of-convex programming. Motivated by recent advances in proximal point methods in both Euclidean and Riemannian settings, we propose two
variants of the proximal point method for solving this class of problems. The first variant requires prior knowledge of the Lipschitz constant of the gradient of the smooth part, making it suitable when this parameter can be readily computed. The second variant, in contrast, does not require such knowledge, thus broadening its applicability. We analyze the complexity of both approaches, establish their convergence, and illustrate their effectiveness through numerical experiments.

To run the adap-GRPPM algorithm, you must provide the functions $g_1$, $g_2$, and $h$ that compose the objective function, along with their corresponding Euclidean gradients (or subgradients).

The numerical examples presented in the manuscript are given below.



### 6.2.1 Sparse Orthogonal Basis Recovery on the $OB(p,n)$ Manifold

The first example is

$$
\min_{X \in OB(p,n)} f_1(X) := g_1(X)+g_2(X)-h(X),
$$

where $g_1(X) = \texttt{tr}(D^4)+\mu \|X\|_1$, $g_2(X) = \texttt{tr}( X^\top A^\top AX)^2$, and $h(X) = 2 \texttt{tr}(D^2X^\top A^\top AX)$.

The codes used in this example are available [here](https://github.com/mbortoloti/AdapGRPPM/tree/main/Example_6.2.1). Please refer to the manuscript for further details. 

### 6.2.2 Sensitivity to the Parameter $\lambda_0$
The second example is 

$$
\min_{P} f_2(X):=g_1(X)+g_2(X)-h(X),
$$

where $g_1(X)= \alpha \texttt{tr}(X)$, $g_2(X) = \texttt{tr}(X^{-1}A) + \log(\det(X))-n$, and $h(X)=\texttt{tr}(BX)$.

The codes used in this example are available [here](https://github.com/mbortoloti/AdapGRPPM/tree/main/Example_6.2.2). Please refer to the manuscript for further details. 


### 6.2.3 Scalability with Manifold Dimension

$$
\min_{P} f_3(X) := g_1(X)+g_2(X)-h(X),
$$

where $g_1(X)=\frac{1}{12} \log(\det(X))^2$, $g_2(X)=\log(\det(X))^3$, and $h(X)=\log(\det(X))$.

### 6.2.4 Comparison with Baseline Methods

$$
\min_P f_4(X) := g_1(X)+g_2(X)-h(X),
$$

where $g_1(X)=\log(\det(X))^4$, $g_2(X)=0$, and $h(X)=\log(\det(X))^2$.

#### Required Julia Packages

The adap-GRPPM algorithm requires the following Julia packages:

* [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/)

* [ManifoldDiff.jl](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/)

* [Manopt.jl](https://manoptjl.org/stable/)

* [LinearAlgebra.jl](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (standard library)

