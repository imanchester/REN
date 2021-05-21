using Flux
using Zygote
using Formatting
using IterTools
using LineSearches

using JuMP

import Base:ndims
# Captures the model state from model return and saves it.
# Returns just the output
function capture_state!(model_output, xt)
    # xt .= (model_output[2])
    copyto!(xt, model_output[2])
    return model_output[1]
end

function my_train!(train_data, val_data, model, opt; Epochs=200)

    ndims(xi::Zygote.Buffer) = ndims(copy(xi))  # ndims not defined for buffer?

    train_batches = size(nth(train_data, 1)[1][1], 2)
    x0 = init_state(model, train_batches)
    
    θ = Flux.trainable(model)    
    ps = Flux.Params(θ)

    sample_loss(y1, y2) = norm(y1 - y2)^2
    loss(y1, y2) = mean(sample_loss.(y1, y2))  

    train_loss_log = []

    for epoch in 1:Epochs
        # Reset the state
        x = x0
        new_x = Zygote.bufferfrom(x)

        batch_loss = []
        max_grads = []
        #  Loop through training data
        for (ut, yt) in train_data
            # ut, yt = nth(train_data, 1)
            # Recur state 
            x = copy(new_x)  
            new_x = Zygote.bufferfrom(x0)  # new buffer to store state
            
            #                                               fancy stuff to recur state 
            train_loss, back = Zygote.pullback(() -> loss(yt, capture_state!(model(x, ut), new_x)), ps)

            append!(batch_loss, train_loss)
            # calculate gradients and update loss
            ∇J = back(one(train_loss))
            update!(opt, ps, ∇J)
            
            # print important training info...=
            grads = Iterators.Filter(!isnothing, ∇J[p] for p in ps)
            g_inf = maximum(norm.(grads))
            append!(max_grads, g_inf)
            printfmt("\r loss: {1:1.2E}", train_loss)
        end
        append!(train_loss_log, mean(batch_loss))
        printfmt("\nEpoch: {1:2d}\tTraining loss: {2:1.4f},\t|g|: {3:1.3E}\n", epoch, mean(batch_loss), maximum(max_grads))

    end
    return train_loss_log
end


function validate(model, val_data, stats; washout=100)
    sample_loss(y1, y2) = norm(y1 - y2)^2
    loss(y1, y2) = mean(sample_loss.(y1, y2))  

    # Allows for case of multiple validation sets.
    res = map(enumerate(val_data)) do (id, (uv, yv)) 
        x0 = init_state(model, size(uv[1], 2))
        yest = model(x0, uv)[1]

        # MSE
        σy, μy = stats.σy, stats.μy
        mse = loss(σy .* yv[washout:end], σy .* yest[washout:end])  

        # NRMSE
        mu = [mean(yv)]
        # nrmse = norm((yv - yest)[washout:end]) / norm(yv[washout:end] .- mu)
        square(x) = x.^2
        dy = (yv - yest)[washout:end]
        nmse = mean(square.(dy)) ./ mean(square.(yv[washout:end] .- mu))
        nrmse = sqrt.(nmse)

        res_i = Dict("u" => uv, "y" => yv, "yest" => yest, "mse" => mse,
                    "nrmse" => nrmse, "washout" => washout)

        return res_i
    end

    return res
end

mse(y1, y2) = mean(norm(y1[:, i] - y2[:, i]).^2 for i in 1:size(y1, 2))
# function train_observer!(model, xn, xt, input_data, opt; Epochs=200, regularizer=nothing, solve_tol=1E-5)
function train_observer!(model, data, opt; Epochs=200, regularizer=nothing, solve_tol=1E-5, min_lr=1E-7)
    θ = Flux.trainable(model)
    ps = Flux.Params(θ)

    mean_loss = [1E5]
    loss_std = []
    for epoch in 1:Epochs
        batch_loss = []
        for (xni, xi, ui) in data
            function calc_loss()
                xpred = model(xi, ui)[1]
                return mean(norm(xpred[:, i] - xni[:, i]).^2 for i in 1:size(xi, 2))
            end

            

            train_loss, back = Zygote.pullback(calc_loss, ps)

            # calculate gradients and update loss
            ∇J = back(one(train_loss))
            update!(opt, ps, ∇J)
        
            push!(batch_loss, train_loss)
            printfmt("Epoch: {1:2d}\tTraining loss: {2:1.4E} \t lr={3:1.1E}\n", epoch, train_loss, opt.eta)
        end

        # Print stats through epoch
        println("------------------------------------------------------------------------")
        printfmt("Epoch: {1:2d} \t mean loss: {2:1.4E}\t std: {3:1.4E}\n", epoch, mean(batch_loss), std(batch_loss))
        println("------------------------------------------------------------------------")
        push!(mean_loss, mean(batch_loss))
        push!(loss_std, std(batch_loss))

        # Check for decrease in loss.
        if mean_loss[end] >= mean_loss[end - 1]
            println("Reducing Learning rate")
            opt.eta *= 0.1
            if opt.eta <= min_lr  # terminate optim.
                return mean_loss, loss_std
            end
        end
    end
    return mean_loss, loss_std
end


function train_observer2!(model, f, g, nx, nu, opt; Epochs=200, regularizer=nothing, solve_tol=1E-5, batchsize=500)
    θ = Flux.trainable(model)
    ps = Flux.Params(θ)

    train_loss_log = []
    for epoch in 1:Epochs

        xt = 1.5 * rand(nx, batchsize) .- 0.2 |> device
        ut = 1.5 * rand(nu, batchsize) .- 0.2 |> device
        xn = f(xt, ut)
        yt = g(xt, ut)
        inputs = vcat(ut, yt)
        function calc_loss()
            xpred = model(xt, inputs)[1]
            return mean(norm(xpred[:, i] - xn[:, i]).^2 for i in 1:size(xt, 2))
        end

        train_loss, back = Zygote.pullback(calc_loss, ps)

        # calculate gradients and update loss
        ∇J = back(one(train_loss))
        update!(opt, ps, ∇J)
        
        append!(train_loss_log, train_loss)
        printfmt("Epoch: {1:2d}\tTraining loss: {2:1.4E}\n", epoch, train_loss)
        if train_loss < solve_tol
            break
        end
    end
end


function snlsdp!(train_data, val_data, model, opt; Epochs=500, β0=10000, patience=20, min_lr=1E-6)
    ndims(xi::Zygote.Buffer) = ndims(copy(xi))  # ndims not defined for buffer?

    batches = size(nth(train_data, 1)[1][1], 2)
    
    β = β0
    x0 = init_state(model, batches)
    θ = [Flux.trainable(model)..., x0]
    ps = Flux.Params(θ)

    best_model = deepcopy(model)

    square_el(M) = M.^2
    # Calc loss including barrier funcctions
    function calc_loss(utild, ytild)
        yest, x = collect(model(x0, utild))
        L = sum(sum(square_el.(yest .- ytild)) / length(ytild))
        LMIs = eval_LMIs(model)
        lmi_sizes = size.(LMIs, 1)  # normalize by lmi size

        ineqs = eval_inequalities(model)
        return L[1] - sum(log.(ineqs)) / length(ineqs) / β  - sum(logdet.(LMIs) ./ lmi_sizes) / β, x[end]
    end

    function check_constraints()
        # Check LMIs are all positive definite
        LMIs = eval_LMIs(model)
        for lmi in LMIs
            try # chol faster than eigs
                cholesky(lmi)  
            catch e
                return false
            end
        end
        # Check inequalities
        ineqs = eval_inequalities(model)
        return all(ineqs .> 0)
    end

    best_loss = Inf

    no_decrease_counter = 0
    vloss_log = [Inf]
    tloss_log = [Inf]
    for epoch in 1:Epochs

        x = x0
        new_x = Zygote.bufferfrom(x)

        # Training loop
        tloss = []
        for (ut, yt) in train_data
            x = copy(new_x)  
            new_x = Zygote.bufferfrom(x0)  # new buffer to store state

            train_loss, back = Zygote.pullback(() -> capture_state!(calc_loss(ut, yt), new_x), ps)
            old_θ = deepcopy(θ)

            ∇J = back(one(train_loss))
            update!(opt, ps, ∇J)  # step parameters

            grads = Iterators.Filter(!isnothing, ∇J[p] for p in ps)
            g_inf = maximum(norm.(grads))

            # backtracking line search
            Δθ = [p2 - p1 for (p1, p2) in zip(old_θ, ps)]
            α = 1
            ls_iters = 1
            valid = check_constraints()
            while ~valid
                α = α / 2
                for (pk, old_θk, Δθk) in zip(ps, old_θ, Δθ)
                    pk .= old_θk + α * Δθk
                end
                valid = check_constraints()

                ls_iters = ls_iters + 1
                if ls_iters > 200
                    break
                end
            end
            # printfmt("\rloss: {1:1.4E}\t ls_iters: {2:d} \t grad: {3:1.3E}", train_loss, ls_iters - 1, g_inf)
            printfmt("\rloss: {1:1.4E}\t ls_iters: {2:d} ", train_loss, ls_iters - 1)
            push!(tloss, train_loss)
        end
        push!(tloss_log, mean(tloss))

        # Check validation set performance
        vloss = []
        vx0 = init_state(model, size(nth(val_data, 1)[1][1], 2))
        for (uv, yv) in val_data
            yest = collect(model(vx0, uv))[1]
            L = sum(sum(square_el.(yest - yv)) / length(yv))
            push!(vloss, L[1])
        end

        if mean(vloss) < best_loss
            best_model = deepcopy(model)
            best_loss = mean(vloss)
        end

        push!(vloss_log, mean(vloss))
        printfmt("\nEpoch: {1:d}\t tloss: {2:1.4E}\t vloss: {3:1.4E}\t lr: {4:1.1E}\t β: {5:1.0E}\n", epoch, mean(tloss), mean(vloss), opt[2].eta, β)

        # Check for sufficient decrease - at least 0.1 percent improvement
        if vloss_log[end] <= minimum(vloss_log[1:end - 1]) - 0.001 * vloss_log[end]
            no_decrease_counter = 0
        else
            no_decrease_counter = no_decrease_counter + 1
        end
            
        if no_decrease_counter > patience
            no_decrease_counter = 0
            println("Decreasing learning rate and increase barrier parameter.")
            β = 100 * β
            opt[2].eta = opt[2].eta / 10

            if opt[2].eta < min_lr
                break
            end
        
        end
    end

    Flux.loadparams!(model, Flux.params(best_model))

    return tloss_log, vloss_log
end

