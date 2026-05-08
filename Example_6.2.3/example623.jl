import Manifolds as mf
import ManifoldDiff as md
using Manopt
import LinearAlgebra as la
using Printf
using Plots
using Random
#using BenchmarkProfiles
using LaTeXStrings
using DelimitedFiles

include("../solvers/adap_rppm_trm.jl")
#include("ppmnn.jl")

NAME_STRING = "EX03_"

# Set dimensions and number of initial guesses (for each dimension) for analysis
n_0 = 10
n_f = 100
Δn = 5
nguess = 1
NAME_STRING = "example623"
# Echo file
#file = open(NAME_STRING * "_ECHO.dat","w")

#################################################################################################
# Warmup
# (only for loading required lybrary) 
#
#################################################################################################

#wg1(M,X) = la.logdet(X)^4
#wgrad_g1(M,X) = 4.0*la.logdet(X)^3 * inv(X)
#
#wg2(M,X) = 0.0
#wgrad_g2(M,X) = zeros(2,2)
#
#wh(M,X) = la.logdet(X)^2
#w∂h(M,X) = 2.0*la.logdet(X) * inv(X)
##w∂H(M,X) = w∂h(X)
#
#wg(M,X) = wg1(M,X) + wg2(M,X)
#wf(M,X) = wg(M,X) - wh(M,X)
#wgrad_g(M,X) = wgrad_g1(M,X) + wgrad_g2(M,X)
#
#
#M = mf.SymmetricPositiveDefinite(2)
#X0 = rand(M)
#λk(n) = sqrt(n)
#adap_rppm(M,X0,wg1,wgrad_g1,wg2,wgrad_g2,wh,w∂h,1.e-2,5,1.e-4); 


dim = [i for i in n_0:Δn:n_f];

seed = MersenneTwister(1234)

global T = Matrix{Float64}(undef,size(dim,1)*nguess,3)
global ntest = 0


for n in dim

    g1(_,X) = la.logdet(X)^4 / 12.0
    grad_g1(_,X) = la.logdet(X)^3 * inv(X) / 3.0


    g2(_,X) = la.logdet(X)^2
grad_g2(_,X) = 2.0 * la.logdet(X) * inv(X)

h(M,X) = la.logdet(X)
∂h(M,X) = la.inv(X)


g(M,X) = g1(M,X) + g2(M,X)
f(M,X) = g(M,X) - h(M,X)
grad_g(M,X) = grad_g1(M,X) + grad_g2(M,X)


for _ in 1:nguess
       
    global ntest += 1
    
    # Set Manifold
    global M = mf.SymmetricPositiveDefinite(n)

    # Set initial guess
    global X0 = log(n)*Matrix{Float64}(la.I,n,n)
    ϵ = 1.e-8
    maxiter = 1000
    ########################################################################
    #                      PPMNN method analysis
    ########################################################################
    try
        λ0 = 1.e-4
        local t0 = time();
        S,error,iter,λk = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ);                
        local et = time() - t0;
        T[ntest,3] = et
        @printf(     "adap-RPPM ::%5d %15.10e %15.10e %+15.10e %8.5e %5d\n",n,et,la.det(S),f(M,S),λk,iter);
 #       @printf(file,"adap-RPPM ::%5d %15.10e %15.10e %+15.10e %8.5e %5d\n",n,et,la.det(S),f(M,S),λk,iter);
        #println("$S")
    catch error
        T[ntest,3] = Inf
        @printf(     "adap-RPPM ::%5d %15.10e\n",n,T[ntest,3]);
        println("$error")
    end
        
    ########################################################################
    #                      End PPMNN analysis
    ########################################################################

end    
end

# Generate and write plot of n x time 
mdim = dim .* ( dim .+ 1.0) ./ 2.0
plot(mdim,T[:,3],label="Adap-RPPM",lw=2,color=:blue)
xlabel!("Manifold Dimension "*L"d = " * "dim " * L"\mathbb{P}^n_{++}")
ylabel!("run time (sec.)")

savefig(NAME_STRING)
