#
#  Code for Example 6.2.1
#
using LinearAlgebra, Manifolds, Random
using Plots,LaTeXStrings

include("../solvers/rpg.jl")
include("../solvers/adap_rppm_sgm.jl")


seed = MersenneTwister(1234)

# dimensões
n = 50         # dimensão da esfera
p = 10        # número de colunas
m = 20          # linhas de A

# dados
A = randn(seed,m, n)

A .-= mean(A,dims=1)

A ./= sqrt.(sum(A.^2,dims=1))

U, S, V = svd(A;full=false)

PCAV = V[:,1:p]
initx = PCAV

tmp = A * PCAV
maxvar = sum(tmp.^2)

#D = Diagonal(S[1:p])

#Dsq = D.^2
Dsq = S[1:p].^2

#lambda = 1 

#Dsq = rand(p)
λ = 1.e-1;
L = norm(Dsq)^2;

tol = 1e-8
maxiter = 10000

# ponto inicial na variedade oblíqua
M = Manifolds.PowerManifold(Sphere(n - 1), p)
X0 = rand(seed,M)

# rodar
Xopt, iter, time, fv, err, sparsity, avar, fs =
    Driver_MPGWH(X0, A, Dsq, λ, L, tol, maxiter);

D2 = Diagonal(Dsq)

g1(_,X) = LinearAlgebra.tr(D2^2) + λ * LinearAlgebra.norm(X,1)

AtA = A'*A

g2(_,X) = LinearAlgebra.tr((X'*AtA*X)^2)

h(_,X) = 2.0 * LinearAlgebra.tr(D2*X'*AtA*X)

grad_g1(_,X) = λ .* sign.(X)

grad_g2(_,X) = 4.0 * AtA * X * X' * AtA * X

grad_h(_,X) = 4.0*AtA*X*D2

M2 = Manifolds.Oblique(n,p)
λ0 = 100.0;
S = adap_rppm(M2,X0,g1,grad_g1,g2,grad_g2,h,grad_h,λ0,maxiter,tol);



plot(fs,xscale = :log10, yscale = :log10,xticks=([1,1e+1,1e+2,1e+3,1e+4],[1,10,100,1000,10000]),lw=2,marker= :square,label="Huang-Wei method",xlabel="Iterations",ylabel=L"$f_1(X^k)$")
plot!(S[5],xscale = :log10, yscale = :log10,lw=2,marker=:circle,label="Adap-GR-PPM")
savefig("example621.png")
