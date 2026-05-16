#
# Code for Example 6.2.2
#

import Manifolds as mf
import ManifoldDiff as md
using Manopt
import LinearAlgebra as la
using Printf
using Plots
using Random
using LaTeXStrings

include("../solvers/adap_grppm_trm.jl")


##################################################################################################

# Set array to storage f(Xk)
cost_adap = []



seed = MersenneTwister(111)

# Set dimension
n = 50

# Identity matrix
In = Matrix{Float64}(la.I,n,n)

# Objective function definitions
α = 0.5
g1(_,X) = α * la.tr(X)
grad_g1(_,X) = α * In
μ = 1.e-3
A = la.Diagonal([i for i in 1:n])
g2(_,X) = la.tr(inv(X)*A)+la.logdet(X) - n
grad_g2(_,X) = -inv(X) * (A * inv(X) - In)
B =  μ * A
h(_,X) = la.tr(B * X)
∂h(_,X) = B

f(M,X) = g1(M,X) + g2(M,X) - h(M,X)

# Set array with test lambda_0's
λ = [1.e-4,1.e-3,1.e-2,1.e-1,1.e+0]

# Set Manifold
M = mf.SymmetricPositiveDefinite(n)

# Set initial guess
X0 = log(n) * Matrix{Float64}(la.I,n,n)

# Calculate f(X0)
f_X0 = f(M,X0)

# Set tolerance
ϵ = 1.e-6

# Set maximum of iterations
maxiter = 1000
            
# λ0 analysis 

λ0 = λ[1]
S = adap_grppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
              return_state=true);
    push!(cost_adap,S)

λ0 = λ[2]
S = adap_grppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
              return_state=true);
    push!(cost_adap,S)

λ0 =λ[3] 
S = adap_grppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
              return_state=true);
    push!(cost_adap,S)

λ0 = λ[4]
S = adap_grppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
              return_state=true);
    push!(cost_adap,S)

λ0 =λ[5] 
S = adap_grppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
              return_state=true);
    push!(cost_adap,S)

# Set critical point for the function above
function xs(i)
    mi = α - μ * i
    return (-1 + sqrt(1 + 4 * mi * i))/(2mi)
end
Xs = la.Diagonal([xs(i) for i in 1:n])


# Ploting  λ0 analysis
plot([],[],
     label="",yticks=([1.e-14,1.e-12,1.e-10,1.e-8,1.e-6,1.e-4,1.e-2,1.e+0]),
     ylabel=L"\log\|f_2(x^k)-f_2(x^*)\|",
     xlabel="Iterations",
     guidefont= font(12),
     tickfont = font(12),
     legendfont = font(12));


# Calculate f(Xs)
f_Xs = f(M,Xs)

for k in 1:size(λ,1)
    it = size(cost_adap[k],1)
    iters = 1:1:it
    l = λ[k]
    label_lambda = @sprintf("%.e",l)
    plot!(iters,abs.(cost_adap[k] .- f_Xs),
          yscale=:log10,lw=2,
          label=L"\lambda_0 = "*"$(label_lambda)")
end

savefig("iter_cost.png")
