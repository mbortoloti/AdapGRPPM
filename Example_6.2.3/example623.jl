#
# Code for Example 6.2.3
#

import Manifolds as mf
import ManifoldDiff as md
using Manopt
import LinearAlgebra as la
using Printf
using Plots
using Random
using BenchmarkTools
using LaTeXStrings
using DelimitedFiles
using Statistics

include("../solvers/adap_grppm_trm.jl")

################################################################################
# Configuration
################################################################################

NAME_STRING = "example623"

n_0   = 10
n_f   = 100
Δn    = 5
nguess = 1

ϵ       = 1.e-8
maxiter = 1000
λ0      = 1.e-4

################################################################################
# Main experiment
################################################################################

function run_experiment()

    dim = collect(n_0:Δn:n_f)

    seed = MersenneTwister(1234)

    # columns:
    # 1 -> n
    # 2 -> manifold dimension
    # 3 -> runtime
    # 4 -> iterations
    # 5 -> memory
    # 6 -> allocations
    T = Matrix{Float64}(undef, length(dim)*nguess, 6)

    ntest = 0

    for n in dim

        ########################################################################
        # Problem definition
        ########################################################################

        g1(_,X) = la.logdet(X)^4 / 12.0
        grad_g1(_,X) = la.logdet(X)^3 * inv(X) / 3.0

        g2(_,X) = la.logdet(X)^2
        grad_g2(_,X) = 2.0 * la.logdet(X) * inv(X)

        h(_,X) = la.logdet(X)
        ∂h(_,X) = inv(X)

        g(M,X) = g1(M,X) + g2(M,X)
        f(M,X) = g(M,X) - h(M,X)

        ########################################################################
        # Experiments
        ########################################################################

        for k in 1:nguess

            ntest += 1

            # Manifold
            M = mf.SymmetricPositiveDefinite(n)

            # Initial point
            X0 = log(n) * Matrix{Float64}(la.I, n, n)

            ####################################################################
            # Warmup (JIT compilation)
            ####################################################################

            try

                S,error,iter,λk = adap_grppm(
                    M,
                    X0,
                    g1,
                    grad_g1,
                    g2,
                    grad_g2,
                    h,
                    ∂h,
                    λ0,
                    maxiter,
                    ϵ
                )

                ################################################################
                # Benchmark
                ################################################################

                bench = @benchmark adap_grppm(
                    $M,
                    copy($X0),
                    $g1,
                    $grad_g1,
                    $g2,
                    $grad_g2,
                    $h,
                    $∂h,
                    $λ0,
                    $maxiter,
                    $ϵ
                ) samples=20 evals=1

                # runtime in seconds
                et = median(bench.times) / 1e9

                # memory and allocations
                mem    = minimum(bench.memory)
                allocs = minimum(bench.allocs)

                ################################################################
                # Save results
                ################################################################

                mdim = n * (n + 1.0) / 2.0

                T[ntest,1] = n
                T[ntest,2] = mdim
                T[ntest,3] = et
                T[ntest,4] = iter
                T[ntest,5] = mem
                T[ntest,6] = allocs

                ################################################################
                # Print
                ################################################################

                @printf(
                    "adap-GR-PPM :: n=%5d time=%12.6e f=%+12.6e λ=%10.4e iter=%5d mem=%10.0f alloc=%10.0f\n",
                    n,
                    et,
                    f(M,S),
                    λk,
                    iter,
                    mem,
                    allocs
                )

            catch err

                T[ntest,1] = n
                T[ntest,2] = n * (n + 1.0) / 2.0
                T[ntest,3] = Inf
                T[ntest,4] = Inf
                T[ntest,5] = Inf
                T[ntest,6] = Inf

                @printf(
                    "adap-GR-PPM :: n=%5d FAILED\n",
                    n
                )

                println(err)

            end

        end
    end

    return T
end

################################################################################
# Run
################################################################################

T = run_experiment()

################################################################################
# Plot
################################################################################

mdim = T[:,2]

p1 = plot(
    mdim,
    T[:,3],
    lw = 2,
    marker = :circle,
    label = "Adap-GR-PPM",
    xlabel = "Manifold Dimension " * L"d = \mathrm{dim}\,\mathbb{P}_{++}^{n}",
    ylabel = "runtime (sec.)",
   # title = "Runtime Analysis",
    legend = :topleft
)

savefig(p1, NAME_STRING * "_runtime.png")

################################################################################
# Save data
################################################################################

writedlm(NAME_STRING * "_runtime.dat", T)

println("\nExperiment finished.\n")
