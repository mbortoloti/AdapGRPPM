

# An adaptative proximal point method for nonsmooth and nonconvex optimization on Hadamard manifolds

## Vitaliano Amaral, Marcio Bortoloti, Jurandir O. Lopes, Gilson Silva


**Abstract** 
This paper addresses a class of nonsmooth and nonconvex optimization problems defined on Hadamard manifolds. The objective function has a composite structure, consisting of convex, differentiable, and lower semicontinuous terms. This formulation may lead to a globally nonconvex problem and encompasses, but is not limited to, the classical difference-of-convex (DC) framework, by allowing more general decompositions beyond the difference of two convex functions.
Motivated by recent advances in proximal point methods in Euclidean and Riemannian settings, we propose two variants: one that uses the Lipschitz constant of the gradient of the smooth part, suitable when this parameter is accessible, and another that does not require such knowledge, thereby broadening its applicability. We analyze the complexity of both approaches, establish their convergence properties, and illustrate their effectiveness through numerical experiments.

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
\min_{X \in P^n_{++}} f_2(X):=g_1(X)+g_2(X)-h(X),
$$

where $g_1(X)= \alpha \texttt{tr}(X)$, $g_2(X) = \texttt{tr}(X^{-1}A) + \log(\det(X))-n$, and $h(X)=\texttt{tr}(BX)$.

The codes used in this example are available [here](https://github.com/mbortoloti/AdapGRPPM/tree/main/Example_6.2.2). Please refer to the manuscript for further details. 


### 6.2.3 Scalability with Manifold Dimension
Next example is

$$
\min_{X \in P^n_{++}} f_3(X) := g_1(X)+g_2(X)-h(X),
$$

where $g_1(X)=\frac{1}{12} \log(\det(X))^2$, $g_2(X)=\log(\det(X))^3$, and $h(X)=\log(\det(X))$.

The codes used in this example are available [here](https://github.com/mbortoloti/AdapGRPPM/tree/main/Example_6.2.3). Please refer to the manuscript for further details. 

For the timing measurements, we used the [BenchmarkTools.jl](https://juliaci.github.io/BenchmarkTools.jl/stable/) package, which provides reliable and accurate benchmarking utilities for Julia implementations.

### 6.2.4 Comparison with Baseline Methods
The last example is

$$
\min_{X \in P^n_{++}} f_4(X) := g_1(X)+g_2(X)-h(X),
$$

where $g_1(X)=\log(\det(X))^4$, $g_2(X)=0$, and $h(X)=\log(\det(X))^2$.

The codes used in this example are available [here](https://github.com/mbortoloti/AdapGRPPM/tree/main/Example_6.2.4). Please refer to the manuscript for further details. 

The DCA and DCPPA methods, available in the [Manopt.jl](https://manoptjl.org/stable/) package, were used exclusively for comparison purposes in the execution time experiments.

For the timing measurements, we used the [BenchmarkTools.jl](https://juliaci.github.io/BenchmarkTools.jl/stable/) package, which provides reliable and accurate benchmarking utilities for Julia implementations.

#### Required Julia Packages for Adap-GR-PPM

The adap-GR-PPM algorithm requires the following Julia packages:

* [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/)

* [ManifoldDiff.jl](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/)

* [LinearAlgebra.jl](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (standard library)

