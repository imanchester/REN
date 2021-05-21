
# Plotting Options
using Printf
using Plots
using LaTeXStrings
using BSON
pgfplotsx()


includet("./models/dense_rren.jl")

theme(:default)

ms = 10
lw = 2.0
line_alpha = 0.8
c = palette(:default);
    
plot_args = (xaxis = :log10, seriestype = :scatter)
xaxis = (L"\gamma", (0, 1000), :log, font(20, "Courier"))

function plot_gamma_vs_perf!(data; kwargs...) 
    plot!([data.gamma_lower], [mean.([data.val[1]["nrmse"]])]; kwargs...)
end

function plot_gamma_vs_perf!(data, gamma_upper; kwargs...) 
    plot!([data.gamma_lower], [mean.([data.val[1]["nrmse"]])]; kwargs..., plot_args...)
    plot!([gamma_upper, gamma_upper], [0, 1.0];  color=kwargs[:color], label=nothing, ls=:dashdot)
end

function format!()
    tfont = font(22, "Courier")
    lfont = font(26, "Courier")

    plot!(;xlabel=L"\underline{\gamma}", xtickfont=tfont, labelfontsize=24, legend=(0.61, 0.99))
    plot!(;ylabel="NRMSE", ytickfont=tfont, legendfontsize=14, framestyle=:box)
end

# ## Plot data F16 Vibration Test
# mf16_ren_4p0 = load_sys_id_test("./results/sys_id/f16/bounded_ren_75_150_4.0")[2]
# mf16_ren_6p0 = load_sys_id_test("./results/sys_id/f16/bounded_ren_75_150_6.0")[2]
# mf16_ren_10p0 = load_sys_id_test("./results/sys_id/f16/bounded_ren_75_150_10.0")[2]
# mf16_ren_20p0 = load_sys_id_test("./results/sys_id/f16/bounded_ren_75_150_20.0")[2]
# mf16_ren_40p0 = load_sys_id_test("./results/sys_id/f16/bounded_ren_75_150_40.0")[2]
# mf16_ren_60p0 = load_sys_id_test("./results/sys_id/f16/bounded_ren_75_150_60.0")[2]
# mf16_ren_100p0 = load_sys_id_test("./results/sys_id/f16/bounded_ren_75_150_100.0")[2]
# mf16_ren_stable = load_sys_id_test("./results/sys_id/f16/stable_ren_75_150")[2]

# # ## Plot data F16 Vibration Test
# mf16_bounded_4p0 = load_sys_id_test("./results/sys_id_cdc2021/f16/bounded_rren_75_150_4.0")[2]
# mf16_bounded_6p0 = load_sys_id_test("./results/sys_id_cdc2021/f16/bounded_rren_75_150_6.0")[2]
# mf16_bounded_10p0 = load_sys_id_test("./results/sys_id_cdc2021/f16/bounded_rren_75_150_10.0")[2]
# mf16_bounded_20p0 = load_sys_id_test("./results/sys_id_cdc2021/f16/bounded_rren_75_150_20.0")[2]
# mf16_bounded_40p0 = load_sys_id_test("./results/sys_id_cdc2021/f16/bounded_rren_75_150_40.0")[2]
# mf16_bounded_60p0 = load_sys_id_test("./results/sys_id_cdc2021/f16/bounded_rren_75_150_60.0")[2]
# mf16_bounded_100p0 = load_sys_id_test("./results/sys_id_cdc2021/f16/bounded_rren_75_150_100.0")[2]
# mf16_stable = load_sys_id_test("./results/sys_id_cdc2021/f16/stable_rren_75_150")[2]

# mf16_ff_4p0 = load_sys_id_test("./results/sys_id/f16/bounded_ffren_75_150_4.0")[2]
# mf16_ff_6p0 = load_sys_id_test("./results/sys_id/f16/bounded_ffren_75_150_6.0")[2]
mf16_ff_10p0 = load_sys_id_test("./results/sys_id/f16/bounded_ffren_75_150_10.0")[2]
mf16_ff_20p0 = load_sys_id_test("./results/sys_id/f16/bounded_ffren_75_150_20.0")[2]
mf16_ff_40p0 = load_sys_id_test("./results/sys_id/f16/bounded_ffren_75_150_40.0")[2]
# mf16_ff_60p0 = load_sys_id_test("./results/sys_id/f16/bounded_ffren_75_150_60.0")[2]
# mf16_ff_100p0 = load_sys_id_test("./results/sys_id/f16/bounded_ffren_75_150_100.0")[2]
mf16_ff_stable = load_sys_id_test("./results/sys_id/f16/stable_ffren_75_150")[2]


# mf16_rrnn_4p0 = load_sys_id_test("./results/sys_id/f16/robust_rnn_lipschitz75_150_4.0")[2]
# mf16_rrnn_6p0 = load_sys_id_test("./results/sys_id/f16/robust_rnn_lipschitz75_150_6.0")[2]
# mf16_rrnn_10p0 = load_sys_id_test("./results/sys_id/f16/robust_rnn_lipschitz75_150_10.0")[2]
# mf16_rrnn_20p0 = load_sys_id_test("./results/sys_id/f16/robust_rnn_lipschitz75_150_20.0")[2]
# mf16_rrnn_40p0 = load_sys_id_test("./results/sys_id/f16/robust_rnn_lipschitz75_150_40.0")[2]
# mf16_rrnn_60p0 = load_sys_id_test("./results/sys_id/f16/robust_rnn_lipschitz75_150_60.0")[2]
# mf16_rrnn_100p0 = load_sys_id_test("./results/sys_id/f16/robust_rnn_lipschitz75_150_100.0")[2]
# mf16_rrnn_stable = load_sys_id_test("./results/sys_id/f16/robust_rnn_stable75_150")[2]

mf16_lstm = load_sys_id_test("./results/sys_id/f16/lstm_170")[2]
mf16_rnn = load_sys_id_test("./results/sys_id/f16/rnn_340")[2]


# gr()
pgfplotsx()
c = palette(:default);
lw = 1.5
plot_font = "Computer Modern"

default(fontfamily=plot_font, labelfontsize=22, legendfontsize=14, tickfontsize=16,
        linewidth=lw, framestyle=:box, label=nothing, grid=true)

p = plot()

plot_gamma_vs_perf!(mf16_ff_10p0, 10.0; label=L"\textrm{R-aREN } \gamma < 10", color=c[2], m=:diamond, msize=ms)
plot_gamma_vs_perf!(mf16_ff_20p0, 20.0; label=L"\textrm{R-aREN }{\gamma <20}", color=c[3], m=:diamond, msize=ms)
plot_gamma_vs_perf!(mf16_ff_40p0, 40.0; label=L"\textrm{R-aREN }{\gamma <40}", color=c[4], m=:diamond, msize=ms)
plot_gamma_vs_perf!(mf16_ff_stable; label=L"\textrm{C-aREN }", color=c[7], m=:diamond, msize=ms)

# Data comes from python impletmentation of Robust RNN
# Data can be found at https://github.com/imanchester/RobustRNN
# branch f16_exp - results/f16
plot!([9.045], [0.5610889]; label=L"\textrm{Robust RNN }{\gamma<10}", color=c[2], m=:dtriangle, msize=ms)
plot!([18.546], [0.475547]; label=L"\textrm{Robust RNN }{\gamma<20}", color=c[3], m=:dtriangle, msize=ms)
plot!([38.846], [0.325848]; label=L"\textrm{Robust RNN }{\gamma<40}", color=c[4], m=:dtriangle, msize=ms)
plot!([148.340], [0.243539]; label=L"\textrm{Robust RNN }{\gamma<\infty}", color=c[7], m=:dtriangle, msize=ms)
# plot!([49.896], [0.335553]; label=L"\textrm{Robust RNN }{\gamma<\infty}", color=c[7], m=:dtriangle, msize=ms)

plot_gamma_vs_perf!(mf16_lstm; label="LSTM", color=c[1], m=:rect, msize=ms)
plot_gamma_vs_perf!(mf16_rnn; label="RNN", color=c[1], m=:circle, msize=ms)

plot!(;xlabel=L"\underline{\gamma}", ylabel="NRMSE", legend=(0.59, 0.98))
ylims!(0.1, 0.60)
minor = [j*10.0^i for i=(0-1):(3+1) for j=2:9]
plot!(; xticks=((10, 100, 1000), ("10", "100", "1000")), minorticks=minor)

savefig(p, "./results/sys_id/f16.pdf")



# plot_gamma_vs_perf!(mf16_lstm; label="LSTM", color=c[1], m=:rect, msize=ms)
# plot_gamma_vs_perf!(mf16_rnn; label="RNN", color=c[1], m=:circle, msize=ms)

# ylims!(0.18, 0.75)
# savefig(p, "./results/sys_id/f16.png")


mean(mf16_ff_stable.val[1]["nrmse"][:])
mean(mf16_bounded_100p0.val[1]["nrmse"][:])

# mf16_bounded_10p0.gamma_lower
# mf16_bounded_20p0.gamma_lower
# mf16_bounded_40p0.gamma_lower
# mf16_bounded_60p0.gamma_lower
# mf16_bounded_100p0.gamma_lower


# mf16_ff_10p0.gamma_lower
# mf16_ff_20p0.gamma_lower
# mf16_ff_40p0.gamma_lower
# mf16_ff_60p0.gamma_lower
# mf16_ff_100p0.gamma_lower
# mf16_ff_stable.gamma_lower


# # Example outputs for f16 dataset & 91.0
# function make_plots()
#     plot_array = Any[]
#     range = 3500:3750
#     for output = 1:3
#         for batch = 1:3
#             y = extract(mf16_bounded_20p0.val[1]["y"], output, batch)[range]
#             yest1 = extract(mf16_bounded_4p0.val[1]["yest"], output, batch)[range]
#             # yest2 = extract(mf16_bounded_10p0.val[1]["yest"], output, batch)[range]
#             yest3 = extract(mf16_bounded_20p0.val[1]["yest"], output, batch)[range]
#             yest6 = extract(mf16_bounded_100p0.val[1]["yest"], output, batch)[range]
#             yest7 = extract(mf16_stable.val[1]["yest"], output, batch)[range]
#             yest8 = extract(mf16_lstm.val[1]["yest"], output, batch)[range]

#             push!(plot_array, plot())
#             plot!(y; c=:black, lw=3)
#             plot!(yest1; c=c[1],lw=1.5)
#             # plot!(yest2; c=c[2],lw=1.5)
#             plot!(yest3; c=c[2], lw=1.5)
#             plot!(yest7; c=c[3], lw=1.5)
#             plot!(yest8; c=c[4], lw=1.5,  legend=nothing)
#         end
#     end
#     return plot(plot_array..., layouts=(3, 3))
# end
# p = make_plots()
# ylims!(-4, 4)
# savefig(p, "./results/sys_id/f16_outputs.png")


# Plot training loss
# c1 = cgrad([:orange, :blue], LinRange(0, 1, 6))
c1 = palette(:thermal, 5)

p = plot(mf16_ff_4p0.tloss, label=L"\textrm{R-aREN }{\gamma < 4}", c=c1[4])
plot!(mf16_ff_10p0.tloss, label=L"\textrm{R-aREN }{\gamma < 10}", c=c1[3])
plot!(mf16_ff_40p0.tloss, label=L"\textrm{R-aREN }{\gamma < 40}", c=c1[2])
plot!(mf16_ff_stable.tloss, label=L"\textrm{C-aREN }", c=c1[1])

plot!(mf16_lstm.tloss, label="LSTM", lw=1.5, c=c[1])
plot!(mf16_rnn.tloss, label="RNN", lw=1.5, c=c[3])
plot!(;xlabel="Epochs", ylabel="Loss" ,yaxis=:log10, legend = (0.66, 0.98))
ylims!(0.5, 11.1)

plot!(;yticks=((1, 10), ("1", "10")))
savefig(p, "./results/sys_id/f16_training.pdf")


# Plot Wiener Hammerstein Results
wh_ff_1p0 = load_sys_id_test("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_1.0")[2]
wh_ff_1p5 = load_sys_id_test("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_1.5")[2]
wh_ff_2p0 = load_sys_id_test("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_2.0")[2]
wh_ff_2p5 = load_sys_id_test("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_2.5")[2]
wh_ff_3p0 = load_sys_id_test("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_3.0")[2]
wh_ff_3p5 = load_sys_id_test("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_3.5")[2]
wh_ff_5p0 = load_sys_id_test("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_5.0")[2]
wh_ff_8p0 = load_sys_id_test("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_8.0")[2]
wh_ff_15p0 = load_sys_id_test("./results/sys_id/wiener_hammerstein/bounded_ffren_40_100_15.0")[2]
wh_ff_stable = load_sys_id_test("./results/sys_id/wiener_hammerstein/stable_ffren_40_100")[2]
wh_lstm = load_sys_id_test("./results/sys_id/wiener_hammerstein/lstm_100")[2]
wh_rnn = load_sys_id_test("./results/sys_id/wiener_hammerstein/rnn_200")[2]



# gr()
pgfplotsx()

p = plot()
plot_gamma_vs_perf!(wh_ff_1p5, 1.5; label=L"\textrm{R-aREN } {\gamma < 1.5}", color=c[2], m=:diamond, msize=ms)
plot_gamma_vs_perf!(wh_ff_2p5, 2.5; label=L"\textrm{R-aREN } {\gamma < 2.5}", color=c[3], m=:diamond, msize=ms)
plot_gamma_vs_perf!(wh_ff_3p5, 3.5; label=L"\textrm{R-aREN } {\gamma < 3.5}", color=c[4], m=:diamond, msize=ms)
plot_gamma_vs_perf!(wh_ff_stable; label=L"\textrm{C-aREN }", color=c[7], m=:diamond, msize=ms)
plot_gamma_vs_perf!(wh_lstm; label="LSTM", color=c[1], m=:utriangle, msize=ms)
plot_gamma_vs_perf!(wh_rnn; label="RNN", color=c[1], m=:circle, msize=ms)

ylims!(0.25, 0.5)
xlims!(1.3, 1000)
plot!(;xlabel=L"\underline{\gamma}", ylabel="NRMSE", legend=(0.64, 0.43))
plot!(; xticks=((10, 100, 1000), ("10", "100", "1000")))
savefig(p, "./results/sys_id/wiener_hammerstein.pdf")


# Plot adversarial perterbations for f16 example
function calc_pert(data; pert_size=0.05)
    model = data.model
    u1 = data.u1
    u2 = data.u2

    du = u2 - u1
    Δu = pert_size * du / norm(du)

    x0 = init_state(model, 1)
    y1 = model(x0, u1)[1]
    y2 = model(x0, u1 + Δu)[1]

    Δy = y2 - y1
    println("Sensitivity: ", norm(Δy) / norm(Δu))

    return Δy, Δu
end

pgfplotsx()
c = palette(:default);
lw = 1.5
plot_font = "Computer Modern"

default(fontfamily=plot_font, labelfontsize=22, legendfontsize=14, tickfontsize=16,
        linewidth=lw, framestyle=:box, label=nothing, grid=true)

ffren_10_dat = BSON.load("./results/sys_id/f16_adversarial_perturbations/ffren_10.bson")[:data]
ffren_40_dat = BSON.load("./results/sys_id/f16_adversarial_perturbations/ffren_40.bson")[:data]
rnn_dat = BSON.load("./results/sys_id/f16_adversarial_perturbations/rnn.bson")[:data]
lstm_dat = BSON.load("./results/sys_id/f16_adversarial_perturbations/lstm.bson")[:data]

rnn_dy, rnn_du = calc_pert(rnn_dat)
lstm_dy, lstm_du = calc_pert(lstm_dat)
ffren_40_dy, ffren_40_du = calc_pert(ffren_40_dat)
ffren_10_dy, ffren_10_du = calc_pert(ffren_10_dat)

k = 1
p=plot(;ylabel=L"\Delta y_1")
plot!(extract(rnn_dy, k, 1); label=L"\textrm{RNN}", c=c[4])
plot!(extract(lstm_dy, k, 1); label=L"\textrm{LSTM}", c=c[3])
plot!(extract(ffren_40_dy, k, 1); label=L"\textrm{R-aREN} \gamma<40", c=c[2])
plot!(extract(ffren_10_dy, k, 1); label=L"\textrm{R-aREN} \gamma<10", c=c[1])
plot!(;xlabel="Time Steps", legend=(0.02, 0.98))
savefig(p, "./results/sys_id/adv_pert.pdf")


# Zoomed in version of the above plot
plot_range = 1000:1500
p=plot(;ylabel=L"\Delta y_1")
plot!(plot_range, extract(rnn_dy[plot_range], k, 1); label=L"\textrm{RNN}", c=c[4])
plot!(plot_range, extract(lstm_dy[plot_range], k, 1); label=L"\textrm{LSTM}", c=c[3])
plot!(plot_range, extract(ffren_40_dy[plot_range], k, 1); label=L"\textrm{R-aREN} \gamma<40", c=c[2])
plot!(plot_range, extract(ffren_10_dy[plot_range], k, 1); label=L"\textrm{R-aREN} \gamma<10", c=c[1])
plot!(;xlabel="Time Steps", legend=(0.01, 0.98))
savefig(p, "./results/sys_id/adv_pert_zoomed.pdf")