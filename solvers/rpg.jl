using Manifolds
using LinearAlgebra
using Printf



function oblique_manifold(n, p)
    return Manifolds.PowerManifold(Sphere(n - 1), p)
end


function f(M, X, A, Dsq, λ)
    AX = A * X
    XtAtAXmD = AX' * AX - Diagonal(Dsq)
    return norm(X, 1) * λ + norm(XtAtAXmD,2)^2
end


function grad_f(M, X, A, Dsq, λ)
    AX = A * X
    XtAtAXmD = AX' * AX - Diagonal(Dsq)
    Eg = 4.0 * A' * (AX * XtAtAXmD)

    # projeção riemanniana automática
    return Manifolds.project(M, X, Eg)
end


function prox_l1_columns(Y, λ)
    Z = max.(abs.(Y) .- λ, 0.0) .* sign.(Y)

    for j in axes(Z, 2)
        if norm(Z[:, j]) > 0
            Z[:, j] /= norm(Z[:, j])
        else
            _, idx = findmax(abs.(Y[:, j]))
            Z[idx, j] = sign(Y[idx, j])
        end
    end
    return Z
end


function MPGWH_solver(
    X0, A, Dsq, λ, L, tol, maxiter;
    c1 = 1e-4,
    maxbtiter = 10
)

    n, p = size(X0)
    M = Manifolds.PowerManifold(Sphere(n - 1), p)

    X = copy(X0)
    iter = 0
    err = Inf
    fs = Float64[]

    # valor inicial
    fX = f(M, X, A, Dsq, λ)
    push!(fs, fX)

    @printf("iter:%d, f:%e\n", iter, fX)

    t0 = Base.time()
    totalbt = 0

    while iter < maxiter #&& err > tol

        # Riemannian gradient
        G = grad_f(M, X, A, Dsq, λ)

        Z = prox_l1_columns(X - G / L, λ / L)

        η = Manifolds.project(M, X, Z - X)

        # busca linear tipo Armijo
        α = 1.0
        btiter = 0

        while btiter < maxbtiter
            Y = Manifolds.retract(M, X, α * η)
            fY = f(M, Y, A, Dsq, λ)

            if fY <= fX - c1 * α * norm(η)^2
                break
            end

            α *= 0.5
            btiter += 1
            totalbt += 1
        end

        if btiter == maxbtiter
            # opcional: warning
            @printf("warning: max backtracking reached\n")
        end

        Y = Manifolds.retract(M, X, α * η)

        dist_XY = Manifolds.distance(M,X,Y)

        if dist_XY < tol
            @printf(
                    "iter:%d, dist(Xk,Xk+1):%15.10f  f:%12.10f\n",
                iter, dist_XY, fY
            )


            return X, iter, Base.time() - t0, fX, err, fs
        end

        fY = f(M, Y, A, Dsq, λ)

        
        #err = norm(α * η * L)^2

        err = norm(α * η)^2
        iter += 1
        push!(fs, fY)

        if iter % 500 == 0
            @printf(
                    "iter:%d, dist(Xk,Xk+1):%15.10f  f:%12.10f\n",
                iter, dist_XY, fY
            )
        end

        X = Y
        fX = fY
    end
   @printf(
                    "iter:%d, dist(Xk,Xk+1):%15.10f  f:%12.10f\n",
                iter, dist_XY, fY
            )

    return X, iter, Base.time() - t0, fX, err, fs
end


function Driver_MPGWH(X0, A, Dsq, λ, L, tol, maxiter)

    Xopt, iter, etime, fv, err, fs =
        MPGWH_solver(X0, A, Dsq, λ, L, tol, maxiter)

    sparsity = count(x -> x == 0.0, Xopt) / length(Xopt)

    Q, R = qr(A * Xopt)
    avar = tr(R' * R)

    return Xopt, iter, etime, fv, err, sparsity, avar, fs
end
