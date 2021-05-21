using Pkg
Pkg.activate(".")

using BenchmarkTools
using Distributions
using Flux.Optimise:update!
using Revise
using Flux.Optimise
using Random
using BSON: @save, @load
using LaTeXStrings
using ColorSchemes
using Match
using Plots

Random.seed!(123);

includet("./models/robust_rnn.jl")
includet("./models/solvers.jl")
includet("./models/REN.jl")
includet("./models/ffREN.jl")
includet("./models/models.jl")
includet("utils.jl")
includet("./data_handler.jl")
includet("train.jl")

plotlyjs()

function save_results(file_name, model, tloss, val_res, gamma, 
                      model_options, data_options, train_options)
    
    set = data_options.set
    path = string("./results/sys_id/", set, "/")

    try
        print("Making directory: ", path, "...")
        mkdir(path)
        println("Done!")
    catch
        println("Directory already exists")
    end


    print("Saving model:", path,  file_name, ".params...")
    @save string(path, file_name, ".params") model
    println("Done")
    
    save_data = (model_options = model_options, 
                 data_options = data_options,
                 train_options = train_options,
                 gamma_lower = gamma,
                 tloss = tloss,
                 val = val_res)

    print("Saving experimental data:", path,  file_name, ".bson...")
    @save string(path, file_name, ".bson") save_data
    println("Done")
end

function run_sys_id_test(file_name, model_options, data_options, train_options)
    
    Random.seed!(123);

    download_and_extract(data_options.set)
    train, val, stats    = @match data_options.set begin
        "cascaded_tanks" => load_cascaded_tanks()
        "silverbox" => load_silverbox()
        "wing_flutter" => load_wing_flutter()
        "f16" => load_f16()
        "wiener_hammerstein" => load_wiener_hammerstein()
        "heat_exchanger" => load_liquid_saturated_heat_exchanger()
    end

    println("Making model: ", model_options...)
    model = @match model_options.model begin
        "bounded_ren" => bounded_ren(model_options.model_args...)
        "stable_ren" => stable_ren(model_options.model_args...)
        "bounded_ffren" => bounded_ff_rnn(model_options.model_args...)
        "stable_ffren" => stable_ff_rnn(model_options.model_args...)
        "lstm" => lstm(model_options.model_args...)
        "rnn" => rnn(model_options.model_args...)
        "robust_rnn_stable" => robust_rnn_stable(model_options.model_args...)
        "robust_rnn_lipschitz" => robust_rnn_lipschitz(model_options.model_args...)
    end

    # Make training and validation datasets
    ut = Iterators.partition(train[1], data_options.seq_len)
    yt = Iterators.partition(train[2], data_options.seq_len)
    train_data = zip(Iterators.partition(train[2], data_options.seq_len))

    train_data = zip(ut, yt) 
    val_data = [val] 

    if model_options.model == "robust_rnn_stable" || model_options.model == "robust_rnn_lipschitz"
        opt = Flux.Optimise.ADAM(train_options.schedule.η)
        tloss, vloss = snlsdp!(train_data, train_data, model, opt; Epochs=train_options.Epochs, β0=1000)    
    else

        # Account for number of steps in epoch
        sched = train_options.schedule
        opt = Flux.Optimiser(Flux.Optimise.ExpDecay(sched.η, sched.decay_rate, 
                                                        length(train_data) * sched.decay_steps, 
                                                        sched.min_lr),
                                    ClipValue(train_options.clip_grad),
                                    Flux.Optimise.ADAM(train_options.schedule.η))

        tloss = my_train!(train_data, val_data, model, opt; Epochs=train_options.Epochs)
    end
    val_res = validate(model, val_data, stats)
    println(val_res[1]["nrmse"])
    gamma = estimate_lipschitz_lower(model; seq_len=200, maxIter=10,step_size=1E-1)
    
    save_results(file_name, model, tloss, val_res, gamma, model_options, data_options, train_options)
    return model, tloss
end

function load_sys_id_test(file_path)
    @load string(file_path, ".params") model
    @load string(file_path, ".bson") save_data
    return model, save_data
end

function recalculate_lipschitz(file_path; step_size=1E-2, clip_at=1E-2, init_var=1E-5)
    # Recalulates Lipschtiz lower bound and saves. 
    model, results = load_sys_id_test(file_path)
    gamma_lower = estimate_lipschitz_lower(model; step_size=step_size, clip_at=clip_at, init_var=init_var, maxIter=300, seq_len=3000)

    save_data = (model_options = results.model_options, 
                 data_options = results.data_options,
                 train_options = results.train_options,
                 gamma_lower = gamma_lower,
                 tloss = results.tloss,
                 val = results.val)

    @save string(file_path, ".bson") save_data
    return gamma_lower
end


# # Silverbox
nx = 10
nv = 40
ϵ = 1E-2
solver = PeacemanRachford(tol=1E-4, α=1.0, maxIter=2000, verbose=false, cg=false)
(model::implicit_rnn)(x0, u) = model(x0, u, solver)  # capture solver

# Silver box experiments
data_options = (seq_len = 1024, set = "silverbox")
train_options = (Epochs = 50, clip_grad = 1E-1, seed = 123,
                    schedule = (η = 1E-3, decay_rate = 0.1, decay_steps = 10, min_lr = 1E-6))

model_options = (model = "stable",
                model_args = (nu = 1, ny = 1, nx = nx, nv = nv))

model_name = string("stable_rren_", nx, "_", nv)
M = run_sys_id_test(model_name, model_options, data_options, train_options)

for γ in [1.0, 2.5, 3.5, 5.0, 10.0]
    model_options = (model = "bounded",
                    model_args = (nu = 1, ny = 1, nx = nx, nv = nv, γ = γ))

    model_name = string("bounded_rren_", nx, "_", nv, "_", γ)
    run_sys_id_test(model_name, model_options, data_options, train_options)
end

model_name = "lstm_30"
model_options = (model = "lstm", model_args = (nu = 1, nv = 30, ny = 1))
run_sys_id_test(model_name, model_options, data_options, train_options)

model_name = "rnn_60"
model_options = (model = "rnn", model_args = (nu = 1, nv = 60, ny = 1))
run_sys_id_test(model_name, model_options, data_options, train_options)

# # F16 
nx = 75
nv = 150
solver = PeacemanRachford(tol=1E-4, α=0.5, maxIter=10000, verbose=false, cg=false)
(model::implicit_param)(x0, u) = model(x0, u, solver)  # capture solver

data_options = (seq_len = 1024, set = "f16")
train_options = (Epochs = 70, clip_grad = 1E-1, seed = 123,
                    schedule = (η = 1E-3, decay_rate = 0.1, decay_steps = 20, min_lr = 1E-6))


model_options = (model = "stable_ffren",
                    model_args = (nu = 2, nx = nx, nv = nv, ny = 3))
model_name = string("stable_ffren_", nx, "_", nv)
run_sys_id_test(model_name, model_options, data_options, train_options)


model_options = (model = "stable_ren",
                model_args = (nu = 2, nx = nx, nv = nv, ny = 3))
model_name = string("stable_ren_", nx, "_", nv)
run_sys_id_test(model_name, model_options, data_options, train_options)


for γ in [4.0, 6.0, 10.0, 20.0, 20.0, 40.0, 60.0, 100.0]
    model_options = (model = "bounded_ren",
                    model_args = (nu = 2, nx = nx, nv = nv, ny = 3, γ = γ))
    model_name = string("bounded_ren_", nx, "_", nv, "_", γ)
    run_sys_id_test(model_name, model_options, data_options, train_options)

    model_options = (model = "bounded_ffren",
    model_args = (nu = 2, nx = nx, nv = nv, ny = 3, γ = γ))
    model_name = string("bounded_ffren_", nx, "_", nv, "_", γ)
    run_sys_id_test(model_name, model_options, data_options, train_options)
end

model_options = (model = "lstm", model_args = (nu = 2, nv = 170, ny = 3))
model_name = string("lstm_", model_options.model_args.nv)
run_sys_id_test(model_name, model_options, data_options, train_options)


train_options = (Epochs = 70, clip_grad = 1E-1, seed = 123,
                    schedule = (η = 1E-4, decay_rate = 0.1, decay_steps = 20, min_lr = 1E-6))

model_options = (model = "rnn", model_args = (nu = 2, nv = 340, ny = 3))
model_name = string("rnn_", model_options.model_args.nv)
run_sys_id_test(model_name, model_options, data_options, train_options)



train_options = (Epochs = 70, clip_grad = 1E-1, seed = 123,
                    schedule = (η = 1E-4, decay_rate = 0.1, decay_steps = 20, min_lr = 1E-6))

model_options = (model = "robust_rnn_stable",
                model_args = (nu = 2, nx = nx, nv = nv, ny = 3))
model_name = string("robust_rnn_stable", nx, "_", nv)
run_sys_id_test(model_name, model_options, data_options, train_options)

for γ in [4.0, 6.0, 10.0, 20.0, 20.0, 40.0, 60.0, 100.0]
    model_options = (model = "robust_rnn_lipschitz",
                    model_args = (nu = 2, nx = nx, nv = nv, ny = 3, γ = γ))
    model_name = string("robust_rnn_lipschitz", nx, "_", nv, "_", γ)
    run_sys_id_test(model_name, model_options, data_options, train_options)
end


# Wiener Hammerstein 
nx = 40
nv = 100
solver = PeacemanRachford(tol=1E-6, α=1.0, maxIter=2000, verbose=false, cg=false)
(model::implicit_rnn)(x0, u) = model(x0, u, solver)  # capture solver

data_options = (seq_len = 512, set = "wiener_hammerstein")
train_options = (Epochs = 60, clip_grad = 1E-1, seed = 123,
                    schedule = (η = 1E-3, decay_rate = 0.1, decay_steps = 40, min_lr = 1E-6))

model_options = (model = "stable_ffren",
                model_args = (nu = 2, nx = nx, nv = nv, ny = 1))
model_name = string("stable_ffren_", nx, "_", nv)
M, tloss = run_sys_id_test(model_name, model_options, data_options, train_options)

for γ in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 8.0]
    model_options = (model = "bounded_ffren",
                    model_args = (nu = 2, nx = nx, nv = nv, ny = 1, γ = γ))

    model_name = string("bounded_ffren_", nx, "_", nv, "_", γ)
    run_sys_id_test(model_name, model_options, data_options, train_options)
end 

model_options = (model = "lstm", model_args = (nu = 2, nv = 100, ny = 1))
model_name = string("lstm_", model_options.model_args.nv)
run_sys_id_test(model_name, model_options, data_options, train_options)

model_options = (model = "rnn", model_args = (nu = 2, nv = 200, ny = 1))
model_name = string("rnn_", model_options.model_args.nv)
run_sys_id_test(model_name, model_options, data_options, train_options)     


train_options = (Epochs = 250, clip_grad = 1E-1, seed = 123,
                    schedule = (η = 1E-3, decay_rate = 0.1, decay_steps = 40, min_lr = 1E-6))

model_options = (model = "robust_rnn_stable",
                model_args = (nu = 2, nx = nx, nv = nv, ny = 1))
model_name = string("robust_rnn_stable", nx, "_", nv)
run_sys_id_test(model_name, model_options, data_options, train_options)

for γ in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 8.0]
    model_options = (model = "robust_rnn_lipschitz",
                    model_args = (nu = 2, nx = nx, nv = nv, ny = 1, γ = γ))
    model_name = string("robust_rnn_lipschitz", nx, "_", nv, "_", γ)
    run_sys_id_test(model_name, model_options, data_options, train_options)
end


recalculate_lipschitz("./results/sys_id/heat_exchanger/regularized_rren_40_40")
recalculate_lipschitz("./results/sys_id/heat_exchanger/stable_rren_40_40")
recalculate_lipschitz("./results/sys_id/heat_exchanger/lstm_65")
recalculate_lipschitz("./results/sys_id/heat_exchanger/rnn_130")

# Heat exchanger
nx = 40
nv = 40
solver = PeacemanRachford(tol=1E-4, α=1.0, maxIter=10000, verbose=false, cg=false)
(model::implicit_rnn)(x0, u) = model(x0, u, solver)  # capture solver

data_options = (seq_len = 50, set = "heat_exchanger")
train_options = (Epochs = 250, clip_grad = 1E-1, seed = 123,
                    schedule = (η = 1E-3, decay_rate = 0.5, decay_steps = 200, min_lr = 1E-4))

model_options = (model = "stable", model_args = (nu = 1, ny = 1, nx = nx, nv = nv))
model_name = string("stable_rren_", nx, "_", nv)
run_sys_id_test(model_name, model_options, data_options, train_options)

model_options = (model = "regularized", model_args = (nu = 1, ny = 1, nx = nx, nv = nv, γ = 2.0))
model_name = string("regularized_rren_", nx, "_", nv)
run_sys_id_test(model_name, model_options, data_options, train_options)

model_options = (model = "lstm", model_args = (nu = 1, nv = 65, ny = 1))
model_name = string("lstm_", model_options.model_args.nv)
run_sys_id_test(model_name, model_options, data_options, train_options)

model_options = (model = "rnn", model_args = (nu = 1, nv = 130, ny = 1))
model_name = string("rnn_", model_options.model_args.nv)
run_sys_id_test(model_name, model_options, data_options, train_options)

#  Run these to recalculate empirical lipschitz lower bounds
recalculate_lipschitz("./results/sys_id/f16/rnn_340")
recalculate_lipschitz("./results/sys_id/f16/lstm_170")

recalculate_lipschitz("./results/sys_id/f16/bounded_rren_75_150_10.0")
recalculate_lipschitz("./results/sys_id/f16/bounded_rren_75_150_20.0")
recalculate_lipschitz("./results/sys_id/f16/bounded_rren_75_150_40.0")
recalculate_lipschitz("./results/sys_id/f16/bounded_rren_75_150_100.0")
recalculate_lipschitz("./results/sys_id/f16/bounded_rren_75_150_60.0")
recalculate_lipschitz("./results/sys_id/f16/stable_rren_75_150")

recalculate_lipschitz("./results/sys_id/f16/bounded_ffren_75_150_10.0")
recalculate_lipschitz("./results/sys_id/f16/bounded_ffren_75_150_20.0")
recalculate_lipschitz("./results/sys_id/f16/bounded_ffren_75_150_40.0")
recalculate_lipschitz("./results/sys_id/f16/bounded_ffren_75_150_100.0")
recalculate_lipschitz("./results/sys_id/f16/bounded_ffren_75_150_60.0")
recalculate_lipschitz("./results/sys_id/f16/stable_ffren_75_150")


recalculate_lipschitz("./results/sys_id/f16/robust_rnn_lipschitz75_150_4.0")
recalculate_lipschitz("./results/sys_id/f16/robust_rnn_lipschitz75_150_6.0")
recalculate_lipschitz("./results/sys_id/f16/robust_rnn_lipschitz75_150_10.0")
recalculate_lipschitz("./results/sys_id/f16/robust_rnn_lipschitz75_150_20.0")
recalculate_lipschitz("./results/sys_id/f16/robust_rnn_lipschitz75_150_40.0")
recalculate_lipschitz("./results/sys_id/f16/robust_rnn_lipschitz75_150_60.0")
recalculate_lipschitz("./results/sys_id/f16/robust_rnn_lipschitz75_150_100.0")
recalculate_lipschitz("./results/sys_id/f16/robust_rnn_stable75_150")




recalculate_lipschitz("./results/sys_id/wiener_hammerstein/rnn_200")
recalculate_lipschitz("./results/sys_id/wiener_hammerstein/lstm_100")
recalculate_lipschitz("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_1.5")
recalculate_lipschitz("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_2.5")
recalculate_lipschitz("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_3.5")
recalculate_lipschitz("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_8.0")
recalculate_lipschitz("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_15.0")
recalculate_lipschitz("./results/sys_id/wiener_hammerstein/stable_ffren_40_100")
