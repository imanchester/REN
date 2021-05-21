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
using Match
using BSON
using DifferentialEquations
using LaTeXStrings

using CUDA

includet("./models/solvers.jl")
includet("./models/REN.jl")
includet("./models/ffREN.jl")
includet("./models/dense_rren.jl")
includet("./train.jl")
includet("./utils.jl")
includet("./PDEs.jl")

plotlyjs()

# Observer design experiment - start with linear system
dtype = Float64
device = cpu

# select to choose model
model_type = "ff_ren"  # "ren", "ff_ren"

ω = 0.0
ν = 0.0

nv = 200
n = 51
m = 1
p = 1


# Generate data
# f, g = reaction_diffusion_equation(;process_noise=0.0, measurement_noise=0.0)
f, g = reaction_diffusion_equation(;process_noise=ω, measurement_noise=ν)

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
plot(X[1, :]; label="1")
# plot!(X[10, :]; label="10")
plot!(X[26, :]; label="26")
plot!(g(X, U)';label="y" )

xt = X[:, 1:end - 1] |> device
xn = X[:, 2:end] |> device
y = g(X, U) |> device

input_data = y[:, 1:end - 1]  # inputs to observer
batchsize = 200

data = Flux.Data.DataLoader((xn, xt, input_data), batchsize=batchsize, shuffle=true)

# Model parameters
nx = n
nu = size(input_data, 1)

# Create implicit model
model = @match model_type begin 
    "ren" => implicit_ren{dtype}(nu, nx, nv;nl=tanh, ϵ=0.01f0, bx_scale=0.0) |> device
    "ff_ren" => implicit_ff_cell{dtype}(nu, nx, nv;nl=tanh, ϵ=0.01f0, bx_scale=0.0) |> device
end

solver = PeacemanRachford(;tol=1E-3, cg=false, verbose=false)
(model::implicit_ren)(x, u) = model(x, u, solver)

# Saving and loading a model
opt = Flux.Optimise.ADAM(1E-3)
tloss, loss_std = train_observer!(model, data, opt; Epochs=50, solve_tol=1E-5)

path = "./results/observer_design/"
model_name = "_nx$(nx)_nv$(nv)_eps_0.01_v0.01"
bson(string(path, model_type, model_name, ".bson"), Dict("model" => cpu(model)))
bson(string(path, model_type, model_name, "_loss", ".bson"), Dict("tloss" => tloss, "tloss_std" => loss_std))


# Test both models
ff_model = BSON.load("./results/observer_design/ff_ren_nx51_nv200_eps_0.01.bson")["model"]
ff_model_noisy = BSON.load("./results/observer_design/ff_ren_nx51_nv200_eps_0.01_v0.01.bson")["model"]


# Test resulting model
T = 2000
time = 1:T

u = ones(dtype, m, length(time)) / 2|> device
x = ones(dtype, n, length(time)) |> device
x2 = zeros(dtype, n, length(time)) |> device

for t in 1:T - 1
    x[:, t + 1] = f(x[:, t:t], u[t:t])
    x2[:, t + 1] = f(x2[:, t:t], u[t:t])
    
    u_next = u[t] + 0.05f0 * (randn(dtype) - 0.01)
    
    if u_next > 1
        u_next = 1
    elseif u_next < 0
        u_next = 0
    end
    u[t + 1] = u_next
end

y = g(x, u) |> device
# y = [g(x[:, t:t], u[1, t]) for t in time] |> device

plot(y[2,:])

# Test Observer
batches = 1
# observer_inputs = [repeat([ui; yi], outer=(1, batches)) for (ui, yi) in zip(u, y)] |> device
observer_inputs = [y[:, t] for t in  1:size(y, 2)]

x0 = zeros(nx, batches) |> device
# xhat_eq = collect(simulate(eq_model, cpu(x0), cpu(observer_inputs), solver))[1]
xhat_ff = collect(simulate(ff_model, cpu(x0), cpu(observer_inputs)))[1]
xhat_ff_noisy = collect(simulate(ff_model_noisy, cpu(x0), cpu(observer_inputs)))[1]

# Xhat_eq = reduce(hcat, xhat_eq)
Xhat_ff = reduce(hcat, xhat_ff)
Xhat_ff_noisy = reduce(hcat, xhat_ff_noisy)


plotlyjs()

default(fontfamily=plot_font, labelfontsize=18, legendfontsize=14, tickfontsize=16,
        linewidth=lw, framestyle=:box, label=nothing, grid=true)

p1 = heatmap(x; color=:thermal, clims=(0.0, 1.0), xticks=nothing, ylabel="True")
p2 = heatmap(Xhat_ff[:, 1:batches:end], color=:thermal, clims=(0.0, 1.0), xticks=nothing,ylabel="Observer")
p3 = heatmap(abs.(x - Xhat_ff[:, 1:batches:end]), color=:thermal, clims=(0.0, 1.0),  ylabel="Error", xlabel = "Time Steps")


p = plot(p1, p2, p3; layout=(3, 1), yticks=((), ()))

xlims!((0.0, 2000.0))
savefig(p,"./results/observer_design/pde_observer.pdf")


## Plot Formatting options
pgfplotsx()
plot_error = true

plot_font = "Computer Modern"
lw = 1.5
default(fontfamily=plot_font, labelfontsize=18, legendfontsize=14, tickfontsize=16,
        linewidth=lw, framestyle=:box, label=nothing, grid=true)


c = palette(:default);
##
for k = 1:1:n
    p1 = plot(x[k,:]; grid=true, c=c[1], label="True")
    p3 = plot!(Xhat_ff[k, :]; c=c[2], label="Observer")
    p2 = plot!(x2[k,:]; c=c[3], label="Simulation")

    plot!(;xlabel="Time Steps", legend=(0.75, 0.98), yaxis=[0,1])
    # plot!(;xlabel="Time Steps", legend=nothing, yaxis=[0,1])
    plot!(;ylabel=L"\xi^{%$k}_t")
    p = plot!()

    # plot_name = "./results/observer_design/state_estimates/rd$(k).pdf"
    plot_name = "./results/observer_design/state_estimates/rd$(k)_legend_on.pdf"
    savefig(p, plot_name)
end



# Plot error
err = Xhat_ff - x
err_sim = x2 - x

error = map(norm, eachslice(err; dims=2))
error_sim = map(norm, eachslice(err_sim; dims=2))

p = plot(error; label="Observer", xlabel="Time Steps", ylabel="Error", c=c[2])
plot!(error_sim; label="Simulation", legend=(0.74, 0.98), c=c[3])

savefig(p,"./results/observer_design/state_estimates/Error.pdf")