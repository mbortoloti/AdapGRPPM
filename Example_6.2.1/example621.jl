#
#  Code for Example 6.2.1
#
using LinearAlgebra, Manifolds, Random
using Plots,LaTeXStrings

include("../solvers/rpg.jl")
include("../solvers/adap_grppm_sgm.jl")


seed = MersenneTwister(1234)

# dimensões
n = 50         
p = 10        
m = 20        

# Set objective function 
A = randn(seed,m, n)

A .-= mean(A,dims=1)

A ./= sqrt.(sum(A.^2,dims=1))

U, S, V = svd(A;full=false)

PCAV = V[:,1:p]
initx = PCAV

tmp = A * PCAV
maxvar = sum(tmp.^2)


Dsq = S[1:p].^2


λ = 1.e-1;
L = norm(Dsq)^2;

tol = 1e-8
maxiter = 10000

# Initial guess on manifold
M = Manifolds.PowerManifold(Sphere(n - 1), p)
X0 = rand(seed,M)

# RPG solver
Xopt, iter, time, fv, err, sparsity, avar, fs =
    Driver_MPGWH(X0, A, Dsq, λ, L, tol, maxiter);

#Set objective function for Adap-GR-PPM

D2 = Diagonal(Dsq)

g1(_,X) = LinearAlgebra.tr(D2^2) + λ * LinearAlgebra.norm(X,1)

AtA = A'*A

g2(_,X) = LinearAlgebra.tr((X'*AtA*X)^2)

h(_,X) = 2.0 * LinearAlgebra.tr(D2*X'*AtA*X)

grad_g1(_,X) = λ .* sign.(X)

grad_g2(_,X) = 4.0 * AtA * X * X' * AtA * X

grad_h(_,X) = 4.0*AtA*X*D2

M2 = Manifolds.Oblique(n,p)

# Set initial λ0
λ0 = 100.0;

# Adap-GR-PPM solver
S = adap_grppm(M2,X0,g1,grad_g1,g2,grad_g2,h,grad_h,λ0,maxiter,tol);



plot(fs,xscale = :log10, yscale = :log10,xticks=([1,1e+1,1e+2,1e+3,1e+4],[1,10,100,1000,10000]),lw=2,marker= :square,label="Huang-Wei method",xlabel="Iterations",ylabel=L"$f_1(X^k)$")
plot!(S[5],xscale = :log10, yscale = :log10,lw=2,marker=:circle,label="Adap-GR-PPM")
savefig("example621l100.png")
