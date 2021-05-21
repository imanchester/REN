using LinearAlgebra
using Flux
using CUDA
import Flux.gpu, Flux.cpu
using Distributions

includet("./solvers.jl")
includet("./fixed_point_layer_general.jl")
includet("./output_layer.jl")

# Implicit, direct parametrizations
abstract type implicit_param end

# Explicit types that we simulate
mutable struct explicit_ren_cell
    ϕ
    A
    B1
    B2
    C1
    D11
    D12
    bx
    bv
end

function (implicit_cell::implicit_param)(xt, ut, solver)
    explicit_cell = explicit(implicit_cell)
    return explicit_cell(xt, ut, solver)
end

# Forward method of rnn-cell
function (model::explicit_ren_cell)(xt, ut, solver::ForwardSolver)
    # First solve fixed point equation for v
    b = model.C1 * xt + model.D12 * ut .+ model.bv
    f(z, x) = model.D11 * model.ϕ.(z) + x
    vt = fixed_point_layer(solver, f, b)
    wt = model.ϕ.(vt)

    # next state and current output
    xn = model.A * xt + model.B1 * wt + model.B2 * ut .+ model.bx
    return xn, (xt, wt)
end

function simulate(explicit_param::explicit_ren_cell, x0, u, solver::ForwardSolver)
    eval_cell = (x, u) -> explicit_param(x, u, solver)
    recurrent = Flux.Recur(eval_cell, x0)
    return unzip(recurrent.(u))
end

function simulate(cell::implicit_param, x0, u, solver::ForwardSolver)
    explicit_param = explicit(cell)
    return simulate(explicit_param, x0, u, solver)
end


function (exp_cell::explicit_ren_cell)(xt, ut, solver::OperatorSplitting, V)
    # First solve fixed point equation for v
    b = exp_cell.C1 * xt + exp_cell.D12 * ut .+ exp_cell.bv
    wt = fixed_point_layer(solver, exp_cell.ϕ, exp_cell.D11, b, V)
    # wt = exp_cell.ϕ.(vt)

    # next state and current output
    xn = exp_cell.A * xt + exp_cell.B1 * wt + exp_cell.B2 * ut .+ exp_cell.bx
    return xn, (xt, wt)
end

function (exp_cell::explicit_ren_cell)(xt, ut, solver::OperatorSplitting)
    # If operator splitting but V is not provided
    V = ((1 + solver.α) * I - solver.α * exp_cell.D11)  
    return exp_cell(xt, ut, solver, V)
end

function simulate(explicit_param::explicit_ren_cell, x0, u, solver::OperatorSplitting, V)
    eval_cell = (x, u) -> explicit_param(x, u, solver, V)
    recurrent = Flux.Recur(eval_cell, x0)
    return unzip(recurrent.(u))
end

function simulate(explicit_param::explicit_ren_cell, x0, u, solver::OperatorSplitting)
    V = ((1 + solver.α) * I - solver.α * explicit_param.D11)
    return simulate(explicit_param, x0, u, solver, V)
end

function simulate(cell::implicit_param, x0, u, solver::OperatorSplitting)
    explicit_param = explicit(cell)
    return simulate(explicit_param, x0, u, solver)
end


## Stable REN cell
mutable struct implicit_ren{T} <: implicit_param
    ϕ
    λ::Union{Vector{T},CuVector{T}}
    V::Union{Matrix{T},CuMatrix{T}}
    S_1::Union{Matrix{T},CuMatrix{T}}
    S_2::Union{Matrix{T},CuMatrix{T}}
    B2::Union{Matrix{T},CuMatrix{T}}
    D12::Union{Matrix{T},CuMatrix{T}}
    bx::Union{Vector{T},CuVector{T}}
    bv::Union{Vector{T},CuVector{T}}
    ϵ::T
end
Flux.trainable(L::implicit_ren) = [L.λ, L.V, L.S_1, L.S_2, L.B2, L.D12, L.bx, L.bv]

function Flux.gpu(M::implicit_ren{T}) where T
    if T != Float32
        println("Moving type: ", T, " to gpu may not be supported. Try Float32!")
    end
    return implicit_ren{T}(M.ϕ, gpu(M.λ), gpu(M.V), gpu(M.S_1), gpu(M.S_2), 
                           gpu(M.B2), gpu(M.D12), gpu(M.bx), gpu(M.bv), M.ϵ)
end

function Flux.cpu(M::implicit_ren{T}) where T
    return implicit_ren{T}(M.ϕ, cpu(M.λ), cpu(M.V), cpu(M.S_1), cpu(M.S_2), 
                           cpu(M.B2), cpu(M.D12), cpu(M.bx), cpu(M.bv), M.ϵ)
end

# TODO: This is currently not great as it does not account for dissipativity in Htild construction
function implicit_ren{T}(nu, nx, nv; nl=relu, ϵ=0.01, bx_scale=0.0, bv_scale=1.0) where T
    glorot_normal(n, m) = convert.(T, randn(n, m) / sqrt(n + m))
    E = Matrix{T}(I, nx, nx)
    F = Matrix{T}(I, nx, nx)
    P = Matrix{T}(I, nx, nx)
    B1 = zeros(T, nx, nv)
    B2 = glorot_normal(nx, nu)
 
    C1 = zeros(T, nv, nx)
    D11 = glorot_normal(nv, nv)
    D12 = zeros(T, nv, nu)

    λ = rand(T, nv)
    Λ = Diagonal(exp.(λ))
    H22 = 2Λ - D11 - D11'
    Htild = [(E + E' - P) -C1' F';
             -C1 H22 B1'
             F B1  P] + ϵ * I

    S_1 = glorot_normal(nx, nx)
    S_2 = glorot_normal(nv, nv)

    V = Matrix{T}(cholesky(Htild).U) # H = V'*V
    
    bv = convert.(T, bv_scale * randn(nv) / sqrt(nv))
    bx = convert.(T, bx_scale * rand(nx) / sqrt(nx))

    return implicit_ren{T}(nl, λ, V, S_1, S_2, B2, D12, bx, bv, ϵ)
end
implicit_ren(nu, nx, nv; nl=relu, ϵ=0.01, bx_scale=0.0, bv_scale=1.0) = implicit_ren{Float64}(nu, nx, nv; nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale)

function explicit(model::implicit_ren)
    nx = size(model.B2, 1)
    nu = size(model.B2, 2)
    nv = length(model.λ)

    H = model.V' * model.V + model.ϵ * I

    # For some reason taking view doesn't work with CUDA
    H11 = H[1:nx, 1:nx]
    H22 = H[ nx + 1:nx + nv, nx + 1:nx + nv]
    H33 = H[ nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]
    H21 = H[nx + 1:nx + nv, 1:nx]
    H31 = H[ nx + nv + 1:2nx + nv, 1:nx]
    H32 = H[ nx + nv + 1:2nx + nv, nx + 1:nx + nv]
    
    # Implicit model parameters
    S_1 = (model.S_1 - model.S_1') / 2    
    S_2 = (model.S_2 - model.S_2') / 2
    
    B2 = model.B2
    D12 = model.D12

    # There appears to be a bug in the zygote adjoint for diagm
    Λᵢ = exp.(model.λ)
    Λ = Diagonal(Λᵢ)

    P = H33
    E = (H11 + P + S_1) / 2
    F = H31

    # equilibrium network stuff
    C1 = - (H21)
    B1 = H32
    D11 = Λ - H22 / 2  - S_2
    
    # Construct explicit rnn model. 
    # Use \bb to differentiate from implicit model   
    𝔸 = E \ F
    𝔹_1 = E \ B1
    𝔹_2 = E \ B2

    ℂ_1 = (1 ./ Λᵢ) .* C1
    𝔻_11 = (1 ./ Λᵢ) .* D11
    𝔻_12 = (1 ./ Λᵢ) .* D12
    
    bx = model.bx
    bv = model.bv

    return explicit_ren_cell(model.ϕ, 𝔸, 𝔹_1, 𝔹_2, ℂ_1, 𝔻_11, 𝔻_12, bx, bv)
end


# Model seq2seq dissipative REN
mutable struct dissipative_ren{T} <: implicit_param
    nu
    nx
    nv
    ny
    implicit_ren::implicit_ren{T}
    output::output{T}
    Q::Matrix{T}
    S::Matrix{T}
    R::Matrix{T}
end

# return paramaters for recurrent and output layers
Flux.trainable(M::dissipative_ren) = (Flux.trainable(M.implicit_ren)..., Flux.trainable(M.output)...)

function dissipative_ren{T}(nu, nx, nv, ny, Q, S, R; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) where T
    ren_params = implicit_ren{T}(nu, nx, nv; nl=nl, ϵ, bx_scale, bv_scale)
    output_params = output{T}(nu, nx, nv, ny)
    return dissipative_ren{T}(nu, nx, nv, ny, ren_params, output_params, Q, S, R)
end

function init_state(model::dissipative_ren{T}, batches) where T
    return typeof(model.implicit_ren.V)(zeros(T, model.nx, batches))
end

function init_state(model::implicit_ren{T}, batches) where T
    nx = size(model.B2, 1)
    return typeof(model.V)(zeros(T, nx, batches))
end

function (model::dissipative_ren)(x0, u, solver::Solver)
    cell = explicit(model)
    # eval_cell = (x, u) -> cell(x, u, solver)
    x, w = simulate(cell, x0, u, solver)
    y = model.output.(x, w, u)
    return y, x[end]
end

function (model::dissipative_ren)(x0, u, solver::OperatorSplitting)
    cell = explicit(model)
    D11 = cell.D11
    V = ((1 + solver.α) * I - solver.α * D11)

    # eval_cell = (xt, ut) -> cell(xt, ut, solver, V) # Capture solver, V
    x, w = simulate(cell, x0, u, solver, V)
    y = model.output.(x, w, u)
    return y, x[end]
end

# Common forms of dissipativity
function bounded_ren(T::DataType, nu, nx, nv, ny, γ; 
                        nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0)
    R = Matrix{T}(γ * I, nu, nu)
    S = zeros(T, nu, ny)
    Q = Matrix{T}(-I / γ, ny, ny)
    return dissipative_ren{T}(nu, nx, nv, ny, Q, S, R; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) 
end

function stable_ren(T::DataType, nu, nx, nv, ny; 
                        nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0)
    return bounded_ren(T, nu, nx, nv, ny, Inf; nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale) 
end

function passive_ren(T::DataType, nu, nx, nv, ny; 
                     η=0.0, nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0)

    R = zeros(T, nu, nu)
    S = Matrix(I / 2, nu, ny)
    Q = Matrix(-η * I, nu, nu)
    return dissipative_ren(nu, nx, nv, ny, Q, S, R; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) 
end

# default type is Float64
bounded_ren(nu, nx, nv, ny, γ; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) = 
                    bounded_ren(Float64, nu, nx, nv, ny, γ; 
                            nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale)


stable_ren(nu, nx, nv, ny; nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) = 
                    stable_ren(Float64, nu, nx, nv, ny; 
                                nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale)

passive_ren(nu, nx, nv, ny; η=0.0, nl=relu, ϵ=0.001, bx_scale=0.0, bv_scale=1.0) = 
                    passive_ren(Float64, nu, nx, nv, ny; 
                                        η=η, nl=nl, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale)
            
                                       
# Construct and explicit 
function explicit(model::dissipative_ren)
    nx = model.nx
    nu = model.nu
    ny = model.ny
    nv = model.nv

    # dissipation parameter
    Q = model.Q
    S = model.S
    R = model.R
    
    # Implicit model parameters
    S_1 = (model.implicit_ren.S_1 - model.implicit_ren.S_1') / 2
    S_2 = (model.implicit_ren.S_2 - model.implicit_ren.S_2') / 2
    
    C2 = model.output.C2
    D21 = model.output.D21
    D22 = model.output.D22

    B2 = model.implicit_ren.B2
    D12 = model.implicit_ren.D12
    

    # There appears to be a bug in the zygote adjoint for diagm
    Λᵢ = exp.(model.implicit_ren.λ)
    Λ = Diagonal(Λᵢ)

    # RHS of dissipation inequality
    Γ1 = [C2'; D21'; zeros(nx, ny)] * Q * [C2 D21 zeros(ny, nx)]  # possibly transpose zeros
    Γ2 = [(C2' * S'); (D21' * S' - D12); B2] * inv(R) * [(S * C2) (S * D21 - D12') B2']

    H = model.implicit_ren.V' * model.implicit_ren.V - Γ1 + Γ2 + model.implicit_ren.ϵ * I 
    H11 = H[1:nx, 1:nx]
    H22 = H[ nx + 1:nx + nv, nx + 1:nx + nv]
    H33 = H[ nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]
    H21 = H[nx + 1:nx + nv, 1:nx]
    H31 = H[ nx + nv + 1:2nx + nv, 1:nx]
    H32 = H[ nx + nv + 1:2nx + nv, nx + 1:nx + nv]

    # Extract parameters
    P = H33
    E = (H11 + P + S_1) / 2
    F = H31


    # equilibrium network stuff
    C1 = -H21
    B1 = H32
    D11 = Λ - H22 / 2  - S_2
    

    # Construct explicit rnn model. Use \bb font to differentiate
    A = E \ F
    𝔹_1 = E \ B1
    𝔹_2 = E \ B2

    ℂ_1 = (1 ./ Λᵢ) .* C1
    𝔻_11 = (1 ./ Λᵢ) .* D11
    𝔻_12 = (1 ./ Λᵢ) .* D12
    
    
    bx = model.implicit_ren.bx
    bv = model.implicit_ren.bv
    
    return explicit_ren_cell(model.implicit_ren.ϕ, A, 𝔹_1, 𝔹_2, ℂ_1, 𝔻_11, 𝔻_12, bx, bv)
end


function check_equilibrium_solve(model, solver::Solver; batches=10)
    explicit_model = explicit(model)

    nx, nu = size(model.B2)
    x0 = randn(nx, batches)
    u = randn(nu, batches)

    xn, (xt, wt) = explicit_model(x0, u, solver)

    bias = explicit_model.C1 * x0 + explicit_model.D12 * u .+ explicit_model.bv

    err = norm(wt - explicit_model.ϕ.(explicit_model.D11 * wt + bias))
    println("Checking tolerance of explicit solve: |w - ϕ(Dw + b)| = ", err)
    println("     Relative tolerance: |w - ϕ(Dw + b)| / |w| = ", err / norm(wt))
end

function check_lmi(model::implicit_ren)
    nx = size(model.B2, 1)
    nu = size(model.B2, 2)
    nv = length(model.λ)

    H = model.V' * model.V + model.ϵ * I

    # For some reason taking view doesn't work with CUDA
    H11 = H[1:nx, 1:nx]
    H22 = H[ nx + 1:nx + nv, nx + 1:nx + nv]
    H33 = H[ nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]
    H21 = H[nx + 1:nx + nv, 1:nx]
    H31 = H[ nx + nv + 1:2nx + nv, 1:nx]
    H32 = H[ nx + nv + 1:2nx + nv, nx + 1:nx + nv]
    
    # Implicit model parameters
    S_1 = (model.S_1 - model.S_1') / 2    
    S_2 = (model.S_2 - model.S_2') / 2
    
    B2 = model.B2
    D12 = model.D12

    # There appears to be a bug in the zygote adjoint for diagm
    Λᵢ = exp.(model.λ)
    Λ = Diagonal(Λᵢ)

    P = H33
    E = (H11 + P + S_1) / 2
    F = H31

    # equilibrium network stuff
    C1 = - (H21)
    B1 = H32
    D11 = Λ - H22 / 2  - S_2

    lmi = [(E + E' - P) (-C1') F';
    (-C1) (2Λ - D11 - D11') B1';
    F B1 P]
end

function check_lmi(model::dissipative_ren)
    nx = model.nx
    nu = model.nu
    ny = model.ny
    nv = model.nv

    # dissipation parameter
    Q = model.Q
    S = model.S
    R = model.R
    
    # Implicit model parameters
    S_1 = (model.implicit_ren.S_1 - model.implicit_ren.S_1') / 2
    S_2 = (model.implicit_ren.S_2 - model.implicit_ren.S_2') / 2
    
    C2 = model.output.C2
    D21 = model.output.D21
    D22 = model.output.D22

    B2 = model.implicit_ren.B2
    D12 = model.implicit_ren.D12

    # RHS of dissipation inequality
    Γ1 = [C2'; D21'; zeros(nx, ny)] * Q * [C2 D21 zeros(ny, nx)]  # possibly transpose zeros
    Γ2 = [(C2' * S'); (D21' * S' - D12); B2] * inv(R) * [(S * C2) (S * D21 - D12') B2']

    H = model.implicit_ren.V' * model.implicit_ren.V - Γ1 + Γ2 + model.implicit_ren.ϵ * I 
    H11 = H[1:nx, 1:nx]
    H22 = H[ nx + 1:nx + nv, nx + 1:nx + nv]
    H33 = H[ nx + nv + 1:2nx + nv, nx + nv + 1:2nx + nv]
    H21 = H[nx + 1:nx + nv, 1:nx]
    H31 = H[ nx + nv + 1:2nx + nv, 1:nx]
    H32 = H[ nx + nv + 1:2nx + nv, nx + 1:nx + nv]

    # Extract parameters    
    Λᵢ = exp.(model.implicit_ren.λ)
    Λ = Diagonal(Λᵢ)
    D11 = Λ - H22 / 2  - S_2
    P = H33
    E = (H11 + P + S_1) / 2
    F = H31
    C1 = -H21
    B1 = H32

    lmi = [(E + E' - P + C2' * Q * C2) (C2' * Q * D21 - C1') (C2' * S') F';
            (D21' * Q * C2 - C1) (2Λ - D11 - D11' + D21' * Q * D21) (D21' * S' - D12) B1';
            (S * C2) (S * D21 - D12') R B2';
            F B1 B2 P]
end

function test_REN()

    nx =  5
    # Check equilibrium solve accuracy
    model = implicit_ren{Float64}(5, 5, 10; nl=tanh)
    solver = PeacemanRachford(;tol=1E-8)
    nPoints = 100

    x0 = init_state(model, nPoints)
    utrain = randn(nx, nPoints)
    ytrain = sin.(utrain)

    xn, (xt, wt) = model(x0, utrain, solver)

    exp_cell = explicit(model)
    b = exp_cell.C1 * x0 + exp_cell.D12 * utrain .+ exp_cell.bv
    err = norm(wt - exp_cell.ϕ.(exp_cell.D11 * wt + b))

    if err > 1E-6
        println("Forward solve doesnt seem to be working well")
    end
    
    # check gradient calculation via FD
    sample_loss(y1, y2) = norm(y1 - y2)^2
    L() = mean(sample_loss.(ytrain, model(x0, utrain, solver)[1]))
    fd_test_grads(L, Flux.params(model))  # param gradients
    fd_test_grads(L, Flux.Params([x0, utrain]))  # input gradients

    # Try simple noiseless regression task
    model = implicit_ren(1, 1, 10; ϵ=1.0, nl=tanh)

    nPoints = 100
    utrain = randn(1, nPoints)
    ytrain = (utrain.^3)
    x0 = init_state(model, nPoints)


    loss() = norm(ytrain - model(x0, utrain, solver)[1])
    ps = Flux.Params(Flux.trainable(model))
    
    opt = Flux.Optimise.ADAM(1E-3)
    for k in 1:1000
        train_loss, back = Zygote.pullback(loss, ps)

        # calculate gradients and update loss
        ∇J = back(one(train_loss))
        update!(opt, ps, ∇J)
        
        printfmt("Iteration: {1:2d}\tTraining loss: {2:1.2E}\n", k, train_loss)
    end

    exp_cell = explicit(model)
    ϕ = exp_cell.ϕ
    W = exp_cell.D11
    b = exp_cell.C1 * x0 + exp_cell.D12 * utrain .+ exp_cell.bv

    xn, (xt, wt) = model(x0, utrain, solver)
    err = norm(wt - ϕ.(W * wt + b))
    zeq = wt

    # Run to test if gradient is wrong
    fd_test_grads(loss, Flux.params(model))  # param gradients
    fd_test_grads(loss, Flux.Params([x0, utrain]))  # input gradients


    yest, (xt, wt) = model(x0, utrain)

    plot(utrain', ytrain'; seriestype=:scatter)
    plot!(utrain', yest'; seriestype=:scatter)

end