using Pkg
Pkg.activate(".")

using BenchmarkTools
using Distributions
using Flux.Optimise:update!
using Revise
using Flux.Optimise
using Random
using Plots
using ControlSystems

using DifferentialEquations

using CUDA

includet("../models/solvers.jl")
includet("../models/REN.jl")
includet("../models/ffREN.jl")
includet("../train.jl")
includet("../utils.jl")
includet("../PDEs.jl")

plotlyjs()

# Observer design experiment - start with linear system
dtype = Float32
device = gpu

nv = 200

n = 51
m = 1
p = 1


# Generate data
f, g = reaction_diffusion_equation()

nPoints = 100000
X = zeros(n, nPoints)
U = zeros(m, nPoints)
for t in 1:nPoints - 1
    X[:, t + 1:t + 1] = f(X[:, t:t], U[:, t:t])
    
    # Calculate next u
    u_next = U[1,t] .+ 0.05f0 * randn(dtype)
    if u_next > 1
        u_next = 1
    elseif u_next < 0
        u_next = 0
    end
    U[:,t + 1] .= u_next
end
xt = X[:, 1:end - 1] |> device
xn = X[:, 2:end] |> device
y = g(X, U) |> device

input_data = [device(U); y][:, 1:end - 1]  # inputs to observer
batchsize = 200

data = Flux.Data.DataLoader((xn, xt, input_data), batchsize=batchsize, shuffle=true)

# Model parameters
nx = n
nu = size(input_data, 1)

# Create implicit model
model = implicit_ff_cell{dtype}(nu, nx, nv; nl=tanh, ϵ=0.01f0, bx_scale=0.0) |> device

# Saving and loading a model
# bson("./results/observer_design/ff_ren_nx51_nv200_eps0.01.bson", Dict("model" => cpu(model)))
# testdata = BSON.load("./results/observer_design/ff_ren_nx51_nv200_eps0.01.bson")  # to load model

opt = Flux.Optimise.ADAM(1E-3)
tloss, loss_std = train_observer!(model, data, opt; Epochs=1000, solve_tol=1E-5)

# Test resulting model
T = 1000
time = 1:T

u = ones(dtype, m, length(time)) / 2 |> device
x = ones(dtype, n, length(time)) |> device

for t in 1:T - 1
    x[:, t + 1] = f(x[:, t:t], u[t:t])
    
    # Calculate next u
    u_next = u[t] + 0.05f0 * (randn(dtype))
    if u_next > 1
        u_next = 1
    elseif u_next < 0
        u_next = 0
    end
    u[t + 1] = u_next
end
y = [g(x[:, t:t], u[t]) for t in time] |> device

# Test Observer
batches = 1
observer_inputs = [repeat([ui; yi], outer=(1, batches)) for (ui, yi) in zip(u, y)] |> device

x0 = zeros(nx, batches) |> device
xhat = collect(simulate(testdata["model"], cpu(x0), cpu(observer_inputs)))[1]

p1 = heatmap(x, color=:cividis, aspect_ratio=1);

Xhat = reduce(hcat, xhat)
p2 = heatmap(Xhat[:, 1:batches:end], color=:cividis, aspect_ratio=1);
p3 = heatmap(abs.(x - Xhat[:, 1:batches:end]), color=:cividis, aspect_ratio=1);

p = plot(p1, p2, p3; layout=(3, 1))

plots = []
for k = 1:6:n
    p = plot(x[k,:])
    push!(plots, p)
    for b in 1:batches
        plot!(extract(xhat, k, b))
    end
    plot!()
end
p = plot(plots...; layout=(3, 3), legend=nothing)



# Plot error
Xhat = reduce(hcat, xhat)
err = Xhat - x
error = map(norm, eachslice(err; dims=2))

p = plot(error; yscale=:log10)
