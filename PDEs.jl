
# Consatructs finite difference approximation of
# nonlinear reaction diffusion equation.

function reaction_diffusion_equation(;L=10.0, steps=5, nx=51, c=1.0, sigma=0.1, process_noise=0.0, measurement_noise=0.0)
    dx = L / (nx - 1)
    dt = sigma * dx^2

    xs = range(0.0, length=nx, stop=L)

    function f(u0, d)
        u = copy(u0)
        un = copy(u0)
        for t in 1:steps
            u = copy(un) 
            # FD approximation of heat equation
            f_local(v) = v[2:end - 1, :] .* (1 .- v[2:end - 1, :]) .* ( v[2:end - 1, :] .- 0.5)
            laplacian(v) = (v[1:end - 2, :] + v[3:end, :] - 2v[2:end - 1, :]) / dx^2
            
            # Euler step for time
            un[2:end - 1, :] = u[2:end - 1, :] + dt * (laplacian(u) + f_local(u) / 2 ) +
                                    process_noise*randn(size(u[2:end - 1, :]))

            # Boundary condition
            un[1:1, :]   = d;
            un[end:end, :] = d;
        end
        return u
    end

    function g(u, d)
        return [d + measurement_noise*randn(1, size(d, 2));
                u[end ÷ 2:end ÷ 2, :] + measurement_noise*randn(1, size(u, 2))]
    end
    return f, g
end

