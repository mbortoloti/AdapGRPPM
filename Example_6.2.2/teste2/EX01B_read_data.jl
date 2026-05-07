using Plots, LinearAlgebra, LaTeXStrings, Printf

# Read data from EX01B_CostAnalysis_ADAP.dat
cost_adp = [parse.(Float64,split(line, ",")) for line in eachline("EX01_CostAnalysisCOST_ADAP.dat")];

# Read data from EX01B_CostAnalysis_DCA.dat
#cost_dca = [parse.(Float64,split(line, ",")) for line in eachline("EX01_CostAnalysisCOST_DCA.dat")];

# Read data from EX01B_CostAnalysis_DCPPA.dat 
#cost_dcppa = [parse.(Float64,split(line, ",")) for line in eachline("EX01_CostAnalysisCOST_DCPPA.dat")];


#Critical point
μ = 1.e-3
α = 0.5
n = 50
λ = [1.e-4,1.e-3,1.e-2,1.e-1,1.e+0]
plot([],[],label="",yticks=([1.e-14,1.e-12,1.e-10,1.e-8,1.e-6,1.e-4,1.e-2,1.e+0]),ylabel=L"\log\|f(x^k)-f(x^*)\|",xlabel="Iteration",guidefont= font(12),tickfont = font(12),legendfont = font(12));
for k in 1:size(λ,1)
    In = Matrix{Float64}(I,n,n)
    v = [i for i in 1:n]
    A = Diagonal(v)
    C = μ * A 
    function xs(i)
        mi = α - μ * i
        return (-1 + sqrt(1 + 4 * mi * i))/(2mi)
    end
    Xs = Diagonal([xs(i) for i in 1:n])
    f(X) = α * tr(X) + tr(inv(X)*A) + logdet(X) -n  - tr(C*X)
    f_Xs = f(Xs)
  #it = size(cost_dcppa[k],1)
  #iters = 1:1:it
  #plot!(iters,abs.(cost_dcppa[k] .- f_Xs),yscale=:log10,marker=:dtriangle,lw=2,label="DCPPA")
  #it = size(cost_dca[k],1)
  #iters = 1:1:it
  #plot!(iters,abs.(cost_dca[k] .- f_Xs),yscale=:log10,marker=:diamond,lw=2,label="DCA")
  it = size(cost_adp[k],1)
  iters = 1:1:it
  l = λ[k]
  label_lambda = @sprintf("%.e",l)
  plot!(iters,abs.(cost_adp[k] .- f_Xs),yscale=:log10,lw=2,label=L"\lambda_0 = "*"$(label_lambda)")
end

savefig("iter_cost.png")
