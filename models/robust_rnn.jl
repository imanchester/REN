using Flux

mutable struct robust_rnn_stable
    nu
    nx
    nv
    ny
    ϕ
    E
    F
    B1
    B2
    C1
    D12
    C2
    D21
    D22
    λ
    P
    bx
    bv
    by
end

function robust_rnn_stable(nu, nx, nv, ny; nl=Flux.relu, T=Float64)
    T = Float64
    
    B1 = randn(nx, nv) / sqrt(nx + nv)
    B2 = randn(nx, nu) / sqrt(2 * (nx + nu))
 
    C1 = randn(nv, nx) / sqrt(nv + nx)
    D12 = zeros(nv, nu)

    λ = ones(nv, 1) # Choose sum abs values to make diagonally dominant
    Λ = 2diagm(λ[:, 1])

    C2 = zeros(ny, nx) / sqrt(ny)
    D21 = zeros(ny, nv) / sqrt(ny)
    D22 = zeros(ny, nu) / sqrt(ny)
    V = randn(2nx, 2nx) / sqrt(2nx)

    H = V' * V + [-C1'; B1] * [-C1 B1'] / 2
    H = (H + H') / 2
    S = randn(nx, nx) / sqrt(nx)

    P = H[nx + 1:end, nx + 1:end]
    E = (H[1:nx, 1:nx] + P) / 2 + S - S'
    F = H[nx + 1:end, 1:nx]

    # Check definiteness of LMI
    # lmi = [(E + E' - P) -C1' F';
    #         -C1 2Λ B1';
    #         F B1 P]
    
    # eigvals(lmi)


    bx = zeros(nx)
    bv = randn(nv) / sqrt(nv)
    by = zeros(ny)

    return robust_rnn_stable(nu, nx, nv, ny, nl, E, F, B1, B2, C1, D12, C2, D21, D22, λ, P, bx, bv, by)

end

Flux.trainable(M::robust_rnn_stable) = [M.E, M.F, M.B1, M.B2, M.C1, M.D12, M.C2,
                                    M.D21, M.D22, M.λ, M.P, M.bx, M.bv, M.by]


init_state(model::robust_rnn_stable, batches) = zeros(model.nx, batches)

function (M::robust_rnn_stable)(x0, u)
    
    E = M.E
    A = E \ M.F
    B1 = E \ M.B1
    B2 = E \ M.B2

    # Λ = Diagonal(M.λ[:, 1])
    Λ = I
    C1 = Λ \ M.C1
    D12 = Λ \ M.D12

    C2 = M.C2
    D21 = M.D21
    D22 = M.D22

    function cell(xt, ut)
        wt = M.ϕ.(C1 * xt + D12 * ut)
        xn = A * xt + B1 * wt + B2 * ut
        yt = C2 * xt + D21 * wt + D22 * ut
        return xn, (yt, xt, wt)
    end
    recurrent_cell  = Flux.Recur(cell, x0)

    return collect(unzip(recurrent_cell.(u)))
end


function eval_LMIs(model::robust_rnn_stable; ϵ=1E-6)

    E = model.E

    # return [ (E + E' - 2ϵ * I) ]

    F = model.F
    B1 = model.B1
    C1 = model.C1
    λ = model.λ
    P = model.P

    Λ = Diagonal(λ[:, 1])

    # Apprently assembly mat all at once mutates some memory
    row1 = [(E + E' - P) -C1' F']
    row2 = [-C1 2Λ B1']
    row3 = [F B1 P]
    mat = [row1; row2; row3]
    
    return [mat + mat' - 2ϵ * I] / 2
end

function eval_inequalities(model::robust_rnn_stable; ϵ=1E-6)
    return model.λ
end

# ------------------------------------------------------------------------------------------------

mutable struct robust_rnn_lipschitz
    nu
    nx
    nv
    ny
    γ
    ϕ
    E
    F
    B1
    B2
    C1
    D12
    C2
    D21
    D22
    λ
    P
    bx
    bv
    by
end

init_state(model::robust_rnn_lipschitz, batches) = zeros(model.nx, batches)

function robust_rnn_lipschitz(nu, nx, nv, ny, γ; nl=tanh)

    F = randn(nx, nx) / sqrt(nx)
    B1 = randn(nx, nv) / sqrt(nx)
    B2 = randn(nx, nu) / sqrt(nx)

    C1 = randn(nv, nx) / sqrt(nv)
    D12 = zeros(nv, nu)

    C2 = randn(ny, nx) / sqrt(ny)
    D21 = randn(ny, nv) / sqrt(ny)
    D22 = randn(ny, nu) / sqrt(ny)

    # Rescale input and output to ensure feasability
    Γy = sum(abs.([C2 D21 D22]), dims=2)
    sfy = γ ./ Γy
    C2 = C2 .* sfy
    D21 = D21 .* sfy
    D22 = D22 .* sfy

    
    Γu = sum(abs.([D12' B2' D22']), dims=2)
    sfu = γ ./ Γu
    D12 = D12 .* sfu'
    B2 = B2 .* sfu'
    D22 = D22 .* sfu'

    # Ensures feasability via diagonal dominance
    p = sum(abs.([F B1 B2]), dims=2)
    P = diagm(p[:, 1])
    λ = sum(abs.([C1 D12 B1' D21']), dims=2) / 2
    Λ = diagm(λ[:, 1])
    e = p + sum(abs.([C1' F' C2']), dims=2)
    E = diagm(e[:, 1]) / 2

    bx = zeros(nx)
    bv = randn(nv) / sqrt(nv)
    by = zeros(ny)

    return robust_rnn_lipschitz(nu, nx, nv, ny, γ,  nl, E, F, B1, B2, C1, D12, C2, D21, D22, λ, P, bx, bv, by)

end

Flux.trainable(M::robust_rnn_lipschitz) = [M.E, M.F, M.B1, M.B2, M.C1, M.D12, M.C2,
                                    M.D21, M.D22, M.λ, M.P, M.bx, M.bv, M.by]


function (M::robust_rnn_lipschitz)(x0, u)
    E = M.E
    A = E \ M.F
    B1 = E \ M.B1
    B2 = E \ M.B2

    Λ = Diagonal(M.λ[:, 1])
    C1 = Λ \ M.C1
    D12 = Λ \ M.D12

    C2 = M.C2
    D21 = M.D21
    D22 = M.D22

    function cell(xt, ut)
        wt = M.ϕ.(C1 * xt + D12 * ut)
        xn = A * xt + B1 * wt + B2 * ut
        yt = C2 * xt + D21 * wt + D22 * ut
        return xn, (yt, xt, wt)
    end
    recurrent_cell  = Flux.Recur(cell, x0)

    return collect(unzip(recurrent_cell.(u)))
end

function eval_inequalities(model::robust_rnn_lipschitz; ϵ=1E-6)
    return model.λ
end


function eval_LMIs(model::robust_rnn_lipschitz; ϵ=1E-6)
    E = model.E

    return [E + E' - 2ϵ * I] /  2

    F = model.F
    B1 = model.B1
    B2 = model.B2
    C1 = model.C1
    D12 = model.D12
    C2 = model.C2
    D21 = model.D21
    D22 = model.D22
    λ = model.λ
    P = model.P

    Λ = Diagonal(λ[:, 1])
    γ = model.γ

    Iu = Matrix(I, model.nu, model.nu)
    Iy = Matrix(I, model.ny, model.ny)

    row1 = [(E + E' - P) -C1' zeros(model.nx, model.nu) F' C2']
    row2 = [-C1 2Λ -D12 B1' D21']
    row3 = [zeros(model.nu, model.nx) -D12' (γ * Iu) B2' D22']
    row4 = [F B1 B2 P zeros(model.nx, model.ny)]
    row5 = [C2 D21 D22 zeros(model.ny, model.nx) (γ * Iy)]
    mat = [row1; row2; row3; row4; row5]

    return [mat + mat' - 2ϵ * I] /  2
end