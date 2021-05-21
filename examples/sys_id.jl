using Pkg
Pkg.activate(".")

using BenchmarkTools
using Flux.Optimise:update!
using Revise
using Flux.Optimise
using Random
using Plots

includet("../models/REN.jl")
includet("../models/ffREN.jl")
includet("../models/solvers.jl")
includet("../models/models.jl")
includet("../models/robust_rnn.jl")

includet("../utils.jl")
includet("../data_handler.jl")
includet("../train.jl")


Random.seed!(123);


# Make a model
nu = 1
ny = 1
nx = 75
nv = 150

# nl(x) = (x + cos(x)*sin(x)) / 2
nl(x) = Flux.relu(x)

# model = stable_ff_rnn(Float64, nu, nx, nv, ny; ϵ=0.01)
# model = robust_rnn_stable(nu, nx, nv, ny; nl=nl)

model = bounded_ren(Float64, nu, nx, nv, ny, 0.25f0; ϵ=0.01)
solver = PeacemanRachford(tol=1E-4, α=1.0, maxIter=2000, verbose=false, cg=false)
(model::dissipative_ren)(x0, u) = model(x0, u, solver)  # capture solver

# download and wing flutter example data
seq_len = 1024
download_and_extract("wing_flutter")
train, val, stats = load_wing_flutter()

# Make training and validation datasets
ut = Iterators.partition(train[1], seq_len)
yt = Iterators.partition(train[2], seq_len)
train_data = zip(Iterators.partition(train[2], seq_len))

train_data = zip(ut, yt) 
val_data = [val] 

opt = Flux.Optimise.ADAM(1E-3)

# Test model on sample input
tloss = my_train!(train_data, val_data, model, opt; Epochs=200)
val_res = validate(model, val_data, stats)

# Try model on example input
x0 = init_state(model, 1)
yest, xend = model(x0, train[1])
plot(extract(yest, 1, 1))
plot!(extract(train[2], 1, 1))

norm(extract(yest, 1, 1)) / norm(extract(train[1], 1, 1))
gamma = estimate_lipschitz_lower(model; seq_len=200, maxIter=150,step_size=1E-1)
