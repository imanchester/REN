using Flux
using Flux:LSTMCell, RNNCell
using Distributions

using Convex

# lstm
mutable struct lstm 
    nu
    nv
    ny
    cell
    output
end

function lstm(nu, nv, ny)    
    cell = LSTMCell(nu, nv)
    output = Dense(nv, ny)
    return lstm(nu, nv, ny, cell, output)
end

function (model::lstm)(x0, u)
    h0 = x0[1:model.nv, :]
    c0 = x0[model.nv + 1:2 * model.nv, :]

    recurrent = Flux.Recur(model.cell, (h0, c0))
    states = recurrent.(u)

    return model.output.(states), vcat(recurrent.state...)
end

function init_state(model::lstm, batches)
    return zeros(2 * model.nv, batches)
end
Flux.trainable(model::lstm) = (Flux.params(model.cell)..., Flux.params(model.output)...)

# rnn
# lstm
mutable struct rnn 
    nu
    nv
    ny
    cell
    output
end

function rnn(nu, nv, ny)    
    cell = RNNCell(nu, nv)
    output = Dense(nv, ny)
    return rnn(nu, nv, ny, cell, output)
end

function (model::rnn)(x0, u)
    recurrent = Flux.Recur(model.cell, x0)
    states = recurrent.(u)
    return model.output.(states), recurrent.state
end
function init_state(model::rnn, batches)
    return zeros(model.nv, batches)
end
Flux.trainable(model::rnn) = (Flux.params(model.cell)..., Flux.params(model.output)...)


# Constant memory RNN

mutable struct mRNNCell
    W
    B
    b
end
function mRNNCell(nu, nv)
    W = randn(nv, nv) / sqrt(nv + nv)
    B = randn(nv, nu) / sqrt(nv + nu)
    b =  randn(nv) / sqrt(nv)
    return mRNNCell(W, B, b)
end

function (M::mRNNCell)(z, u)
    xn = M.W * z + M.B * u .+ M.b
    J = [Diagonal(xn[:, i] .>= 0.0) for i in 1:batches]
    α = norm.(J[i] * M.W for i in 1:batches)[:, :]'
    zn = Flux.relu.(xn) ./ α 
    return res, res
end
Flux.trainable(M::mRNNCell) = (M.W, M.B, M.b)

mutable struct mrnn
    nu 
    nv
    ny
    cell
    output
end

function mrnn(nu, nv, ny)    
    cell = mRNNCell(nu, nv)
    output = Dense(nv, ny)
    return mrnn(nu, nv, ny, cell, output)
end

function (model::mrnn)(x0, u)
    recurrent = Flux.Recur(model.cell, x0)
    states = recurrent.(u)
    return model.output.(states), recurrent.state
end

function init_state(model::mrnn, batches)
    return zeros(model.nv, batches)
end
Flux.trainable(model::mrnn) = (Flux.params(model.cell)..., Flux.params(model.output)...)


# LTI Models 
mutable struct LTI
    A
    B
    C
    D
end

function simulate_states(model::LTI, x0, u)
    x = zeros(length(x0), length(u))   
    x[:, 1] = x0
    for tt in 1:length(u)-1
        x[:, tt+1] = model.A*x[:, tt] + model.B*u[tt]
    end
    return x
end

function (model::LTI)(x0, u)
    states = simulate_states(model, x0, u)
    output(xt, ut) = model.C*xt + model.D*ut
    return output.(states, u)
end



mutable struct explicit_robust_rnn
    A
    B1
    B2
    C1
    D12
    bv
    C2
    D21
    D22
    by
end

mutable struct implicit_robust_rnn
    E
    B1
    B2
    C1
    D12
    bv
    C2
    D21
    D22
    by
    Λ
    P
end

function project_stable(exp_model::explicit_robust_rnn)
    𝔸 = exp_model.A
    𝔹1 = exp_model.B1
    𝔹2 = exp_model.B2
    ℂ1 = exp_model.C1
    𝔻12 = exp_model.D12
    ℂ2 = exp_model.C2
    𝔻21 = exp_model.D21
    𝔻22 = exp_model.D22

    Λ = Diagonal(Variable(nv))
    P = Variable(nx, nx)
    E = Variable(nx, nx)

    F = Variable(nx, nx)
    B1 = Variable(nx, nv)
    B2 = Variable(nx, nu)

    C1 = Variable(nv, nx)
    D12 = Variable(nv, nu)

    P = Variable(nx, nx)
    Λ = Diagonal(Variable(nv))

    # Correctness
    error1 = E * [𝔸 𝔹1 𝔹2] - [F B1 B2]
    error2 = Λ * [ℂ1 𝔻12] - [C1 D12]
    J = norm(error1) + norm(error2)

    # Stability constraint
    M = [E+E'-P -C1' F'
         -C1 2Λ B1'
         F B1 P]

    constraints = [M⪰1E-4]
    solver = () -> SCS.Optimizer(verbose=1)
    problem = minimize(γ, Constraints...)
    Convex.solve!(problem, solver)
end


function make_robust_rnn_plot()
    
end