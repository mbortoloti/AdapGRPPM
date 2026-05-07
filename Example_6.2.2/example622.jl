import Manifolds as mf
import ManifoldDiff as md
using Manopt
import LinearAlgebra as la
using Printf
using Plots
using Random
#using BenchmarkProfiles
using LaTeXStrings
#using DelimitedFiles

include("../solvers/adap_rppm_trm.jl")
#include("ppmnn.jl")

# Put here the experiment identification
NAME_STRING = "EX01_"

# Set dimensions and number of initial guesses (for each dimension) for analysis
#n_0 = parse(Int64,ARGS[1])
#n_f = parse(Int64,ARGS[2])
#Δn = parse(Int64,ARGS[3])
#nguess = parse(Int64,ARGS[4])
#ExpId = ARGS[5]

ExpId = "TESTE"

NAME_STRING = NAME_STRING * ExpId

##################################################################################################
cost_adap = []
cost_DCA  = []
cost_DCPPA= []

#dim = [i for i in n_0:Δn:n_f];

seed = MersenneTwister(111)

#global T = Matrix{Float64}(undef,size(dim,1)*nguess,3)
global ntest = 0


#for n in dim

    n = 10

    In = Matrix{Float64}(la.I,n,n)

    α = 0.5
    g1(_,X) = α * la.tr(X)
    grad_g1(_,X) = α * In

 #   μ = 1.e-3
    μ = 2.0
    #A = 3.0 * In
    A = la.Diagonal([i for i in 1:n])
    g2(_,X) = la.tr(inv(X)*A)+la.logdet(X) - n
    #In = Matrix{Float64}(la.I,n,n)
    grad_g2(_,X) = -inv(X) * (A * inv(X) - In)

    #C = rand(seed,n,n)
    #Q,_ = la.qr(C)
    #v = 1.e-3 * [i for i in 1:n]
    #C = la.Diagonal(v) 

    B = μ * A

    h(_,X) = la.tr(B * X)
    ∂h(_,X) = B


    g(M,X) = g1(M,X) + g2(M,X)
    f(M,X) = g(M,X) - h(M,X)
    #grad_g(M,X) = grad_g1(M,X) + grad_g2(M,X)


    #for _ in 1:nguess
       
            global ntest += 1
    
            # Set Manifold
            global M = mf.SymmetricPositiveDefinite(n)

            # Set initial guess
            #global X0 = mf.rand(seed,M)
            global X0 = log(n) * Matrix{Float64}(la.I,n,n)

            f_X0 = f(M,X0)

            ϵ = 1.e-6
            maxiter = 1000
            
            λ0 = 1.e-4
            S = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
            return_state=true);
            push!(cost_adap,S)

            λ0 = 1.e-3
            S = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
            return_state=true);
            push!(cost_adap,S)

            λ0 = 1.e-2
            S = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
            return_state=true);
            push!(cost_adap,S)

            λ0 = 1.e-1
            S = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
            return_state=true);
            push!(cost_adap,S)

            λ0 = 1.0
            S = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
            return_state=true);
            push!(cost_adap,S)

            #λ0 = 1.e+1
            #S = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
            #return_state=true);
            #push!(cost_adap,S)

            #λ0 = 1.e+2
            #S = adap_rppm(M,X0,g1,grad_g1,g2,grad_g2,h,∂h,λ0,maxiter,ϵ;
            #return_state=true);
            #push!(cost_adap,S)



#
#            Rt = difference_of_convex_algorithm(M,f,g,∂h,X0;grad_g=grad_g,
#            debug = [:Iteration," ", (:Cost,"%20.15e"), 1, "\n", :Stop],
#            record = [:Cost], 
#            return_state = true,
#            stopping_criterion=StopAfterIteration(maxiter)|StopWhenChangeLess(M,ϵ),
#            sub_stopping_criterion=StopAfterIteration(5000)|StopWhenGradientNormLess(1.e-10)
#            );

 #           fcost = get_record(Rt)
 #           pushfirst!(fcost,f_X0)
 #           push!(cost_DCA,fcost)
 #           
 #           Rt = nothing
 #           GC.gc()


#            λ(_) = 0.5
#            Rt = difference_of_convex_proximal_point(M,∂h,X0;g=g,grad_g=grad_g,λ=λ,
#            debug = [:Iteration," ",(:Cost,"%20.15e"),"\n", :Stop],cost=f,
#            record= [:Cost],
#            return_state = true,
#            stopping_criterion=StopAfterIteration(maxiter)|StopWhenChangeLess(M,ϵ),
#            sub_stopping_criterion=StopAfterIteration(5000)|StopWhenGradientNormLess(1.e-10)
#            );
            
#            fcost = get_record(Rt)
#            pushfirst!(fcost,f_X0)
#            push!(cost_DCPPA,fcost)
            
#            Rt = nothing
#            GC.gc()


    #end    
#end

open(NAME_STRING * "COST_ADAP.dat","w") do io
    for v in cost_adap
        println(io,join(v,","))
    end
end


#open(NAME_STRING * "COST_DCA.dat","w") do io
#    for v in cost_DCA
#        println(io,join(v,","))
#    end
#end

#open(NAME_STRING * "COST_DCPPA.dat","w") do io
#    for v in cost_DCPPA
#        println(io,join(v,","))
#    end
#end

