using Test
using ITensors
ITensors.disable_warn_order()
using MSSTA

@testset "binaryop.jl" begin
    @testset "_binaryop" for rev_carrydirec in [true, false], nbit in 2:3
        # For a = +/- 1, b = +/- 1, c = +/- 1, d = +/- 1,
        # x' = a * x + b * y
        # y' = c * x + d * y
        # f(x, y) = g(x', y')
        # excluding a + b == -2 || c + d == -2
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
            if a + b == -2 || c + d == -2
                continue
            end
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

            if ! (g_vec_ref ≈ g_vec)
                @show a, b, c, d, bc_x, bc_y
                exit()
            end
            @test g_vec_ref ≈ g_vec
        end
    end

    @testset "binaryop_three_sites" for rev_carrydirec in [true], bc_x in [1], bc_y in [1], bc_z in [1]
        # x' = c1 * x + c2 * y
        # y' =          c3 * y + c4 * z
        # z' = c6 * x          + c5 * z
        # f(x, y, z) = g(x', y', z')
        nbit = 3

        if rev_carrydirec
            # x1, y1, z1, x2, y2, z2, ...
            sites = [Index(2, "Qubit, $name=$n") for n in 1:nbit for name in ["x", "y", "z"]]
        else
            # xR, yR, zR, xR-1, yR-1, zR-1...
            sites = [Index(2, "Qubit, $name=$n") for n in reverse(1:nbit)
                     for name in ["x", "y", "z"]]
        end
        # x1, x2, ...
        sitesx = [sites[findfirst(x -> hastags(x, "x=$n"), sites)] for n in 1:nbit]
        # y1, y2, ...
        sitesy = [sites[findfirst(x -> hastags(x, "y=$n"), sites)] for n in 1:nbit]
        # z1, z2, ...
        sitesz = [sites[findfirst(x -> hastags(x, "z=$n"), sites)] for n in 1:nbit]

        rsites = reverse(sites)

        for coeffs in Iterators.product(fill(collect(-1:1), 6)...)
            any([sum(coeffs[2i-1:2i]) for i in 1:3] .== -2) && continue

            #for coeffs in [(1, -1, 1, 1, 1, 1)]
            M = MSSTA.binaryop_mpo(
                sites,
                [Tuple(coeffs[1:2]), Tuple(coeffs[3:4]), Tuple(coeffs[5:6])],
                [(1, 2), (2, 3), (3, 1)];
                rev_carrydirec=rev_carrydirec, bc=[bc_x, bc_y, bc_z])

            f = randomMPS(sites)
            g = apply(M, f)

            # f[x_R, ..., x_1, y_R, ..., y_1, z_R, ..., z_1]
            f_arr = Array(reduce(*, f), vcat(reverse(sitesx), reverse(sitesy), reverse(sitesz)))

            # g[x'_R, ..., x'_1, y'_R, ..., y'_1, z'_R, ..., z'_1]
            g_arr = Array(reduce(*, g), vcat(reverse(sitesx), reverse(sitesy), reverse(sitesz)))

            # f[x, y, z]
            f_vec = reshape(f_arr, 2^nbit, 2^nbit, 2^nbit)

            # g[x', y', z']
            g_vec = reshape(g_arr, 2^nbit, 2^nbit, 2^nbit)

            function prime_xy(x, y, z)
                xp_ = coeffs[1] * x + coeffs[2] * y
                yp_ =                 coeffs[3] * y + coeffs[4] * z
                zp_ = coeffs[6] * x                 + coeffs[5] * z
                xp = mod(xp_, 2^nbit)
                yp = mod(yp_, 2^nbit)
                zp = mod(zp_, 2^nbit)
                return xp, yp, zp,
                       xp == xp_ ? 1 : bc_x,
                       yp == yp_ ? 1 : bc_y,
                       zp == zp_ ? 1 : bc_z
            end

            g_vec_ref = similar(g_vec)
            for x in 0:(2^nbit - 1), y in 0:(2^nbit - 1), z in 0:(2^nbit - 1)
                xp, yp, zp, sign_x, sign_y, sign_z = prime_xy(x, y, z)
                g_vec_ref[x + 1, y + 1, z + 1] = f_vec[xp + 1, yp + 1, zp+1] * sign_x * sign_y * sign_z
            end

            @test g_vec_ref ≈ g_vec
        end
    end

end

