"""
    adap_grppm(...)

Implementation of the adaptive GR-PPM solver for optimization problems on Hadamard manifolds of the form

    min_{X ∈ M} g₁(X) + g₂(X) - h(X),

where `M` is a Hadamard manifold, `g₁` is a proper and lower semicontinuous function, `g₂` is continuously diffewrentiable with a Lipschitz continuous gradient, and `h` convex.

This solver is based on the adaptive proximal point framework proposed in:

    Vitaliano S. Amaral,
    Marcio Antônio de A. Bortoloti,
    Jurandir O. Lopes,
    Gilson N. Silva,
    "An Adaptive Proximal Point Method for Nonsmooth and Nonconvex Optimization on Hadamard Manifolds".

The implementation provided in this repository is intended to reproduce the numerical experiments and computational results presented in the paper.

This particular implementation employs the subgradient method (available in the Julia package [Manopt.jl](https://manoptjl.org/)) as the internal subsolver, since we are specifically investigating a nonsmooth optimization setting.
"""


import Manifolds as mf
import ManifoldDiff as md
using Manopt
using LinearAlgebra

function build_fk(M :: mf.AbstractManifold, Y :: Matrix{Float64}, Zk :: Matrix{Float64}, 
        g1 :: Function, λk :: Float64)

    return g1(M,Y) + 0.5 * λk * mf.distance(M,Y,Zk)^2

end

function build_gk(M,Y,Zk,λk)

    return 0.5 * λk * mf.distance(M,Y,Zk)^2

end

function build_grad_gk(M,Y,Zk,λk)

    return -λk * mf.log(M,Y,Zk)

end

###########################################################################################################
function build_grad_fk(M :: mf.AbstractManifold, Y :: Matrix{Float64}, Zk :: Matrix{Float64}, 
        grad_g1 :: Function, λk :: Float64)

   # r_grad_g1_Y = Y * grad_g1(M,Y) * Y
    #r_grad_g1_Y = md.riemannian_gradient(M,Y,grad_g1(M,Y))

    r_grad_g1_Y = md.project(M,Y,grad_g1(M,Y))


    return r_grad_g1_Y - λk * mf.log(M,Y,Zk)

end
###########################################################################################################
function adap_grppm(M :: mf.AbstractManifold, X0 :: Matrix{Float64}, g1 :: Function, grad_g1 :: Function, 
        g2 :: Function, grad_g2 :: Function, h :: Function, grad_h :: Function, λ0 :: Float64,
        maxiter :: Int,ϵ :: Float64;return_state = false)

    # Set objective function
    f(M,X) = g1(M,X) + g2(M,X) - h(M,X)
    
    # Set initial guess
    Xk = copy(M,X0)
    println("iter: ",0,"  ","f: ",f(M,Xk))
    f_Xk = f(M,Xk)
  
    # Set λk
    λk = λ0
        
    cost = []
    push!(cost,f_Xk)

    for iter in 1:maxiter

        # Calculate riemannian gradient of g2
        #Vk = md.riemannian_gradient(M,Xk,grad_g2(M,Xk))
        Vk = mf.project(M,Xk,grad_g2(M,Xk))

        # Calculate riemannian subgradient of h
        #Wk = md.riemannian_gradient(M,Xk,grad_h(M,Xk))
        Wk = mf.project(M,Xk,grad_h(M,Xk))

        WkmVk = Wk-Vk 

        while true
            
            Arg_exp = WkmVk / λk
            Zk = mf.exp(M,Xk,Arg_exp)

            # Build functions for subproblem
            fk(M,Y) = build_fk(M,Y,Zk,g1,λk)
            grad_fk(M,Y) = build_grad_fk(M,Y,Zk,grad_g1,λk)
            
            gk(M,Y) = build_gk(M,Y,Zk,λk)

            grad_gk(M,Y) = build_grad_gk(M,Y,Zk,λk)
            try
                # Solving subproblem
                #Yk = Manopt.trust_regions(M, fk, grad_fk, Zk;
                     #stepsize=ArmijoLinesearch(M;contraction_factor=0.5,initial_stepsize=1.0,
                     #stop_when_stepsize_less=1.e-16),
                     #debug=[:Iteration, :Cost, "\n",:Stop],
                #     stopping_criterion=StopAfterIteration(5000)|StopWhenChangeLess(M,1e-12)
                #    );
                stepsize_opt = Manopt.DecreasingStepsize(M::AbstractManifold;
                               length=min(injectivity_radius(M)/2, 1.0),
                               factor=0.9,
                               subtrahend=1.e-5,
                               exponent=2.0,
                               shift=0.0,
                               type=:relative
                               )
                Yk = Manopt.subgradient_method(M, fk, grad_fk, Zk;
                #stepsize = Manopt.DecreasingStepsize(M),
                stepsize = stepsize_opt,
                retraction_method = ProjectionRetraction(),
                stopping_criterion=Manopt.StopAfterIteration(50)
                )
                #length=min(injectivity_radius(M)/2, 1.0),
                #factor=1.5,
                #subtrahend=0.0,
                #exponent=0.5,
                #shift=0.0,
                #type=:relative)
 
                #Yk = Manopt.subgradient_method(M,fk,grad_fk,Xk;
                #    #stepsize = DecreasingStepsize(0.5, 1, 0.5), # (comprimento inicial, expoente, etc)
                #    stopping_criterion = StopAfterIteration(500),
                #    return_state = true)

                #stepsize = Manopt.DecreasingStepsize(M),
                #stepsize = ArmijoLinesearch(M;contraction_factor=0.5,initial_stepsize=1.0,stop_when_stepsize_less=1.e-16),
                #debug = [:Iteration, :Cost],
                #stopping_criterion = Manopt.StopAfterIteration(50000))

                #Yk = Manopt.proximal_gradient_method(M, fk, gk, grad_gk,Xk)
                
                #Yk = Manopt.convex_bundle_method(M, fk, grad_fk, Xk);

                # Only for avoid numerical issues
                Yk = Manifolds.project(M,Yk)

                #debug=[:Iteration, :Cost, "\n",:Stop])
                

                #Yk = Manopt.proximal_gradient_method(M,fk,gk,grad_gk,Xk;debug=[:Iteration, :Cost, "\n",:Stop])
                #;
                #        stepsize = 1.0,stopping_criterion = StopWhenGradientNormLess(1e-6) | StopAfterIteration(5000),
                #        debug = []
                #        )

                if ~(Manifolds.is_point(M,Yk))
                    println("Yk is not on manifold!")
                end

                global dist_Xk_Yk = mf.distance(M,Xk,Yk)

                f_Yk = f(M,Yk)
                #push!(cost,f_Yk)
                # Teste
                #η = Manifolds.project(M,Xk,Yk-Xk)
                #println("norm η = $(norm(η,2)^2)")

                if dist_Xk_Yk < ϵ
                    
                    println("iter: ",iter,"  ","dist(Xk,Xk+1):",dist_Xk_Yk,"  ","f: ",f_Yk,"   ","λk: ",λk)
                    
                    if return_state
                        return cost
                    else
                        return Yk,0,iter,λk,cost,dist_Xk_Yk
                    end
                end

                if ~(f_Yk - f_Xk > -0.25 * λk * dist_Xk_Yk^2)
                    Xk = Yk
                    f_Xk = f_Yk
                    break
                else
                    λk = 2.0 * λk
                end
             #end
          catch  error
            println("$error")
            return nothing
            #λk = 2.0 * λk
            λk = min(2.0 * λk, 1e6)
          end

        end

        push!(cost,f_Xk)

        # Print info
        if mod(iter,500) == 0
            println("iter: ",iter,"  ","f: ",f_Xk,"  ","λk: ",λk)
        end
        
    end
    
    return Xk,-1,maxiter,λk,cost,dist_Xk_Yk

    
end
