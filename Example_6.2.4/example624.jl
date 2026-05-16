#
# Code for Example 6.2.4
#

import Manifolds as mf
import ManifoldDiff as md
using Manopt
import LinearAlgebra as la
using Printf
using Plots
using Random
using BenchmarkProfiles
using LaTeXStrings
using DelimitedFiles
using BenchmarkTools

include("../solvers/adap_grppm_trm.jl")
#include("ppmnn.jl")

#NAME_STRING = "EX02_"

# Set dimensions and number of initial guesses (for each dimension) for analysis
#n_0 = parse(Int64,ARGS[1])
n_0 = 5
#n_f = parse(Int64,ARGS[2])
n_f = 20
#Δn = parse(Int64,ARGS[3])
Δn = 1
#nguess = parse(Int64,ARGS[4])
nguess = 10
#NAME_STRING = NAME_STRING * ARGS[5]
NAME_STRING = "teste"
# Echo file
#file = open(NAME_STRING * "_ECHO.dat","w")

#################################################################################################
# Warmup
#################################################################################################

wg1(M,X) = la.logdet(X)^4
wgrad_g1(M,X) = 4.0*la.logdet(X)^3 * inv(X)

wg2(M,X) = 0.0
wgrad_g2(M,X) = zeros(2,2)

wh(M,X) = la.logdet(X)^2
w∂h(M,X) = 2.0*la.logdet(X) * inv(X)

wg(M,X) = wg1(M,X) + wg2(M,X)
wf(M,X) = wg(M,X) - wh(M,X)
wgrad_g(M,X) = wgrad_g1(M,X) + wgrad_g2(M,X)

Mwarm = mf.SymmetricPositiveDefinite(2)
Xwarm = rand(Mwarm)

# Warmup adap-RPPM
adap_grppm(
    Mwarm,Xwarm,
    wg1,wgrad_g1,
    wg2,wgrad_g2,
    wh,w∂h,
    1.e-2,5,1.e-4
)

# Warmup DCA
Manopt.difference_of_convex_algorithm(
    Mwarm,wf,wg,w∂h,Xwarm;
    grad_g=wgrad_g,
    stopping_criterion=StopAfterIteration(3)|StopWhenChangeLess(Mwarm,1.e-3),
    sub_stopping_criterion=StopAfterIteration(3)|StopWhenGradientNormLess(1.e-3)
)

# Warmup DCPPA
λwarm(k) = 1/(2.0*k)

Manopt.difference_of_convex_proximal_point(
    Mwarm,w∂h,Xwarm;
    g=wg,
    grad_g=wgrad_g,
    λ=λwarm,
    stopping_criterion=StopAfterIteration(3)|StopWhenChangeLess(Mwarm,1.e-3),
    sub_stopping_criterion=StopAfterIteration(3)|StopWhenGradientNormLess(1.e-3)
)

#################################################################################################

dim = [i for i in n_0:Δn:n_f];

seed = MersenneTwister(1234)

global T = Matrix{Float64}(undef,size(dim,1)*nguess,3)
global ntest = 0


for n in dim

g1(M,X) = la.logdet(X)^4
grad_g1(M,X) = 4.0*la.logdet(X)^3 * inv(X)

g2(M,X) = 0.0
grad_g2(M,X) = zeros(n,n)

h(M,X) = la.logdet(X)^2
∂h(M,X) = 2.0*la.logdet(X) * inv(X)

g(M,X) = g1(M,X) + g2(M,X)
f(M,X) = g(M,X) - h(M,X)
grad_g(M,X) = grad_g1(M,X) + grad_g2(M,X)

for _ in 1:nguess
       
    global ntest += 1
    
    # Set Manifold
    global M = mf.SymmetricPositiveDefinite(n)

    # Set initial guess
    global X0 = mf.rand(seed,M)

    ϵ = 1.e-8
    maxiter = 1000

    ########################################################################
    #                      Adap-GRPPM analysis
    ########################################################################

    try
        λ0 = 1.e-4

        println("Adap-GR-PPM n: $n")


        bench = @benchmark adap_grppm(
            $M,$X0,
            $g1,$grad_g1,
            $g2,$grad_g2,
            $h,$∂h,
            $λ0,$maxiter,$ϵ
        ) samples=5 evals=1

        et = minimum(bench).time / 1e9

#        S,error,iter,λk = adap_rppm(
#            M,X0,
#            g1,grad_g1,
#            g2,grad_g2,
#            h,∂h,
#            λ0,maxiter,ϵ
#        )

        T[ntest,3] = et

#        @printf(
#            "adap-RPPM ::%5d %15.10e %15.10e %+15.10e %8.5e %5d\n",
#            n,et,la.det(S),f(M,S),λk,iter
#        )

#        @printf(
#            file,
#            "adap-RPPM ::%5d %15.10e %15.10e %+15.10e %8.5e %5d\n",
#            n,et,la.det(S),f(M,S),λk,iter
#        )

    catch error

        T[ntest,3] = Inf

#        @printf(
#            "adap-RPPM ::%5d %15.10e\n",
#            n,T[ntest,3]
#        )

#        @printf(
#            file,
#            "adap-RPPM ::%5d %15.10e\n",
#            n,T[ntest,3]
#        )

        println(error)
    end
        
    ########################################################################
    #                      DCA analysis
    ########################################################################

    try
 println("DCA  n: $n")
        bench = @benchmark Manopt.difference_of_convex_algorithm(
            $M,$f,$g,$∂h,$X0;
            grad_g=$grad_g,
            stopping_criterion=StopAfterIteration($maxiter)|StopWhenChangeLess($M,$ϵ),
            sub_stopping_criterion=StopAfterIteration(1000)|StopWhenGradientNormLess(1.e-10)
        ) samples=5 evals=1

        et = minimum(bench).time / 1e9

#        S = Manopt.difference_of_convex_algorithm(
#            M,f,g,∂h,X0;
#            grad_g=grad_g,
#            stopping_criterion=StopAfterIteration(maxiter)|StopWhenChangeLess(M,ϵ),
#            sub_stopping_criterion=StopAfterIteration(1000)|StopWhenGradientNormLess(1.e-10)
#        )

        T[ntest,2] = et
#
#        @printf(
#            "DCA       ::%5d %15.10e %15.10e %+15.10e\n",
#            n,et,la.det(S),f(M,S)
#        )

#        @printf(
#            file,
#            "DCA       ::%5d %15.10e %15.10e %+15.10e\n",
#            n,et,la.det(S),f(M,S)
#        )

    catch

        T[ntest,2] = Inf

 #       @printf(
 #           "DCA       ::%5d %15.10e\n",
 #           n,T[ntest,2]
 #       )

 #       @printf(
 #           file,
 #           "DCA       ::%5d %15.10e\n",
 #           n,T[ntest,2]
 #       )
    end
        
    #######################################################################
    #                      DCPPA analysis
    #######################################################################

    try
 println("DCPPA n: $n")
        λ(_) = 1/(2.0*n)

        bench = @benchmark Manopt.difference_of_convex_proximal_point(
            $M,$∂h,$X0;
            g=$g,
            grad_g=$grad_g,
            λ=$λ,
            stopping_criterion=StopAfterIteration($maxiter)|StopWhenChangeLess($M,$ϵ),
            sub_stopping_criterion=StopAfterIteration(1000)|StopWhenGradientNormLess(1.e-10)
        ) samples=5 evals=1

        et = minimum(bench).time / 1e9

 #       S = Manopt.difference_of_convex_proximal_point(
 #           M,∂h,X0;
 #           g=g,
 #           grad_g=grad_g,
 #           λ=λ,
 #           stopping_criterion=StopAfterIteration(maxiter)|StopWhenChangeLess(M,ϵ),
 #           sub_stopping_criterion=StopAfterIteration(1000)|StopWhenGradientNormLess(1.e-10)
 #       )

        T[ntest,1] = et

  #      @printf(
  #          "DCPPA     ::%5d %15.10e %15.10e %+15.10e\n",
  #          n,et,la.det(S),f(M,S)
  #      )

#        @printf(
#            file,
#            "DCPPA     ::%5d %15.10e %15.10e %+15.10e\n",
#            n,et,la.det(S),f(M,S)
#        )

    catch

        T[ntest,1] = Inf

   #     @printf(
   #         "DCPPA     ::%5d %15.10e\n",
   #         n,T[ntest,1]
   #     )

#        @printf(
#            file,
#            "DCPPA     ::%5d %15.10e\n",
#            n,T[ntest,1]
#        )
    end

    #######################################################################
    #                    End of DCPPA analysis
    #######################################################################

end

end

#close(file)

############################################################################
# Performance Profile
############################################################################

ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"

performance_profile(
    PlotsBackend(),
    T,
    ["DCPPA","DCA","Adap-GR-PPM"],
    leg=:bottomright,
    l=2,
    ylabel="Solved Problems (%)",
    xlabel="Performance ratio: CPU time"
)

savefig("example624")

############################################################################
# Dimension x Time plot
############################################################################

#mdim = dim .* (dim .+ 1.0) ./ 2.0
#
#plot(
#    mdim,
#    T[:,1],
#    label="DCPPA",
#    lw=2,
#    xticks=(
#        [0.0,50.0,100.0,150.0,200.0],
#        ["0","50","100","150","200"]
#    )
#)

#plot!(mdim,T[:,2],label="DCA",lw=2)
#plot!(mdim,T[:,3],label="Adap-RPPM",lw=2)

#xlabel!(
#    "Manifold Dimension "*L"d = " * "dim " * L"\mathbb{P}^n_{++}"
#)

#ylabel!("run time (sec.)")

#if nguess == 1
#    savefig(NAME_STRING * "DvT" * "Adap-RPPM")
#end

############################################################################
# Save timing matrix
############################################################################

#writedlm(NAME_STRING * "time_PP.csv",T,',')
