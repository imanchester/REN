using Pkg
Pkg.activate(".")

using Distributions
using LinearAlgebra
using ControlSystems
using JuMP
using Convex
using BSON: @save, @load

using MosekTools
using Mosek

using Plots
using StatsPlots
plotlyjs()

includet("./utils.jl")
includet("./models/ffREN.jl")

# First experiment
η = 1E-1
nx = 50
nv = 500


# Sample simple discrete time system
# Pole locations in polar coordinates
ρ = 0.80
ϕ = 2 * pi / 10
λ = [ρ * cos(ϕ) + ρ * sin(ϕ) * im, ρ * cos(ϕ) - ρ * sin(ϕ) * im]

sys = zpk([], λ, 0.3, 1.0)

Pdy(d) = lsim(sys, d, 1:size(d, 1))[1]
Puy(d) = lsim(sys, d, 1:size(d, 1))[1]

# Sample disurbance and simulate response to that distribution
amp = 10
samples, hold = 500, 50
# samples, hold = 200, 50
function sample_disturbance()
    d = 2 * amp * (rand(samples, 1) .- 0.5)
    d = kron(d, ones(hold, 1))
    return d
end
d = sample_disturbance()

pgfplotsx()
c = palette(:default);
lw = 1.5
plot_font = "Computer Modern"

default(fontfamily=plot_font, labelfontsize=22, legendfontsize=18, tickfontsize=16,
        linewidth=2, framestyle=:box, label=nothing, grid=true)

p=plot(d[1:1000]; label="Example Disturbance", xlabel="Time Steps", legend=(0.51, 0.99))
savefig(p, "./results/youla_example_disturbance.pdf")

# Simulate system with disturbance
b = Pdy(d)

plot(d)
plot!(b)

# Sample ESN and simulate effect of output 
# Q_param = implicit_ff_cell(1, nx, nv)
Q_param = sample_ff_ren(1, nx, nv)
function Q(v)
    x0 = init_state(Q_param, size(v, 2))
    xt, wt = simulate(Q_param, x0, v)

    X = reduce(hcat, xt)
    W = reduce(hcat, wt)
    
    return [X' W' v ones(size(v, 1), 1)]   # include ones for bias term
end

C = Q(-b)
A = reduce(hcat, Puy(ci) for ci in eachslice(C, dims=2))

# Optimize!
θv = Convex.Variable(size(A, 2))

y = A * θv + b
u = C * θv

J = norm(y, 1) + η * norm(θv, 2) + 1E-3*sumsquares(u)
constraints = [u < 5.0, u > -5.0]

problem = minimize(J, constraints)
Convex.solve!(problem, Mosek.Optimizer)

θ = evaluate(θv)

# linear system
λ = 1 - 1E-1
q = 2nx + nv
A = randn(q, q) / sqrt(q)
ρ = maximum(abs.(eigvals(A)))
A = λ * A / ρ
B = randn(q, 1) / sqrt(q)
function Q_lin(v)
    X = zeros(size(v, 1), size(A, 1))
    for t in 1:size(v, 1) - 1
        X[t + 1, :] = A * X[t,:] + B * v[t,:]
    end
    return X
end

Clin = Q_lin(-b)
Alin = reduce(hcat, Puy(ci) for ci in eachslice(Clin, dims=2))

# Optimize!
θv = Convex.Variable(size(Alin, 2))

y = Alin * θv + b
u = Clin * θv

J = norm(y, 1) + η * norm(θv, 2)
constraints = [u < 4.0, u > -4.0]

problem = minimize(J, constraints)
Convex.solve!(problem, Mosek.Optimizer)

θlin = evaluate(θv)

# Test input
amps = range(0, length=7, stop=8)
d_test = reduce(hcat, a .* [ones(1, 50) zeros(1, 50)] for a in amps)'

btest = Pdy(d_test)
Ctest = Q(-btest)
Atest = reduce(hcat, Puy(ci) for ci in eachslice(Ctest, dims=2))

Ctest_lin = Q_lin(-btest)
Atest_lin = reduce(hcat, Puy(ci) for ci in eachslice(Ctest_lin, dims=2))


# Resulting signals
ytest = Atest * θ + btest
utest = Ctest * θ

ytest_lin = Atest_lin * θlin + btest
utest_lin = Ctest_lin * θlin

# --------------------------- Plotting ----------------------------------------------------
pgfplotsx()
c = palette(:default);
lw = 1.5
plot_font = "Computer Modern"

default(fontfamily=plot_font, labelfontsize=22, legendfontsize=18, tickfontsize=16,
        linewidth=2, framestyle=:box, label=nothing, grid=true)

T0 = 25

plot(d_test[T0:end]; label="Disturbance", lw=2.0)
plot!(Pdy(d_test)[T0:end]; label="Open Loop", lw=lw)
plot!(ytest_lin[T0:end]; label="Linear", lw=lw, c=c[4])
plot!(ytest[T0:end]; label="aREN", lw=lw, c=c[3])
plot!(;xlabel="Time Steps", legend=(0.02, 0.98))
p = plot!()
savefig(p, "./results/youla/disturbance_rejection.pdf")

# Control input figure

p = plot(utest_lin[T0:end]; label="Linear", lw=lw, c=c[4])
p = plot!(utest[T0:end]; label="aREN", lw=lw, c=c[3])
plot!([0, length(utest) - T0], [-5, -5]; label="Contraints", c=:black, ls=:dash, lw=2.0)
plot!(;xlabel="Time Steps", legend=(0.02, 0.98))
savefig(p, "./results/youla/control.pdf")

