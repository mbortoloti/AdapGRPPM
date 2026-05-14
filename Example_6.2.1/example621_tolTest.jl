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

Set_tol = [1.e-1,1.e-2,1.e-3,1.e-4,1.e-5,1.e-6,1.e-7,1.e-8]

#tol = 1e-8
maxiter = 100000

# Initial guess on manifold
M = Manifolds.PowerManifold(Sphere(n - 1), p)
X0 = rand(seed,M)

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


iter_adap_gr_ppm = []
iter_rpg = []


for i in 1:size(Set_tol,1)

    tol = Set_tol[i]

# RPG solver
 R = Driver_MPGWH(X0, A, Dsq, λ, L, tol, maxiter);
# println("R=$(R[2])")
 push!(iter_rpg,R[2])

# Set initial λ0
λ0 = 10.0;

# Adap-GR-PPM solver
local S = adap_grppm(M2,X0,g1,grad_g1,g2,grad_g2,h,grad_h,λ0,maxiter,tol);
#println("S=$(S[3])")
push!(iter_adap_gr_ppm,S[3])

end


plot(Set_tol,iter_rpg,xlabel=L"$\texttt{dist}(X_k,X_{k+1})$",ylabel="Iterations",xscale=:log10,yscale=:log10,xticks=(Set_tol,Set_tol),yticks=([10,100,1000,10000,100000],[10,100,1000,10000,100000]),lw = 2,marker=:square,label="Whang-Wei Method")
plot!(Set_tol,iter_adap_gr_ppm,xscale=:log10,yscale=:log10,xticks=(Set_tol,Set_tol),lw=2,marker=:circle,label="Adap-GR-PPM")

#plot(fs,xscale = :log10, yscale = :log10,xticks=([1,1e+1,1e+2,1e+3,1e+4],[1,10,100,1000,10000]),lw=2,marker= :square,label="Huang-Wei method",xlabel="Iterations",ylabel=L"$f_1(X^k)$")
#plot!(S[5],xscale = :log10, yscale = :log10,lw=2,marker=:circle,label="Adap-GR-PPM")
savefig("tol_test_L10.png")
#

