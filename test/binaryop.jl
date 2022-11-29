using Test
using ITensors
using MSSTA

@testset "arithmetic.jl" begin @testset "_binaryop" for rev_carrydirec in [true, false]
    # x' = a * x + b * y
    # y' = c * x + d * y
    # f(x, y) = g(x', y')
    nbit = 2
    nsite = 2nbit
    if rev_carrydirec
        # x1, y1, x2, y2, ...
        sites = [Index(2, "Qubit, $name=$n") for n in 1:nbit for name in ["x", "y"]]
    else
        # xR, yR, xR-1, yR-1, ...
        sites = [Index(2, "Qubit, $name=$n") for n in reverse(1:nbit)
                 for name in ["x", "y"]]
    end
    # x1, x2, ...
    sitesx = [sites[findfirst(x -> hastags(x, "x=$n"), sites)] for n in 1:nbit]
    # y1, y2, ...
    sitesy = [sites[findfirst(x -> hastags(x, "y=$n"), sites)] for n in 1:nbit]
    rsites = reverse(sites)

    for a in -1:1, b in -1:1, c in -1:1, d in -1:1, bc_x in [1, -1], bc_y in [1, -1]
        M = MSSTA.binaryop_mpo(sites, [(a, b), (c, d)], [(1, 2), (1, 2)];
                               rev_carrydirec=rev_carrydirec, bc=[bc_x, bc_y])

        f = randomMPS(sites)
        g = apply(M, f)

        # f[x_R, ..., x_1, y_R, ..., y_1]
        f_arr = Array(reduce(*, f), vcat(reverse(sitesx), reverse(sitesy)))

        # g[x'_R, ..., x'_1, y'_R, ..., y'_1]
        g_arr = Array(reduce(*, g), vcat(reverse(sitesx), reverse(sitesy)))

        # f[x, y]
        f_vec = reshape(f_arr, 2^nbit, 2^nbit)

        # g[x', y']
        g_vec = reshape(g_arr, 2^nbit, 2^nbit)

        function prime_xy(x, y)
            xp_ = a * x + b * y
            yp_ = c * x + d * y
            xp = mod(xp_, 2^nbit)
            yp = mod(yp_, 2^nbit)
            return xp, yp,
                   xp == xp_ ? 1 : bc_x,
                   yp == yp_ ? 1 : bc_y
        end

        g_vec_ref = similar(g_vec)
        for x in 0:(2^nbit - 1), y in 0:(2^nbit - 1)
            xp, yp, sign_x, sign_y = prime_xy(x, y)
            g_vec_ref[x + 1, y + 1] = f_vec[xp + 1, yp + 1] * sign_x * sign_y
        end

        @test g_vec_ref ≈ g_vec
    end
end end
