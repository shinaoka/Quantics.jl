using Test
using ITensors
using MSSTA

@testset "arithmetic.jl" begin
    #=
    @testset "_adder" begin
        link_in = Index(2, "link_in")
        link_out = Index(2, "link_out")
        site_a = Index(2, "site_a")
        site_b = Index(2, "site_b")
        site_out = Index(2, "site_out")
        adder = MSSTA._adder_single_tensor(link_in, link_out, site_a, site_b, site_out)
        @test size(adder) == (2, 2, 2, 2, 2)
        for (cin, a, b, out, cout) in [
                (0, 0, 0, 0, 0),
                (0, 0, 1, 1, 0),
                (0, 1, 0, 1, 0),
                (0, 1, 1, 0, 1),
                (1, 0, 0, 1, 0),
                (1, 0, 1, 0, 1),
                (1, 1, 0, 0, 1),
                (1, 1, 1, 1, 1),
            ]
            @test adder[cin + 1, cout+1, a+1, b+1, out+1] == 1.0
        end
    end
    =#

    @testset "_binaryop" for rev_carrydirec in [true, false]
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
            sites = [Index(2, "Qubit, $name=$n") for n in reverse(1:nbit) for name in ["x", "y"]]
        end
        # x1, x2, ...
        sitesx = [sites[findfirst(x -> hastags(x, "x=$n"), sites)] for n in 1:nbit]
        # y1, y2, ...
        sitesy = [sites[findfirst(x -> hastags(x, "y=$n"), sites)] for n in 1:nbit]
        rsites = reverse(sites)

        for a in -1:1, b in -1:1, c in -1:1, d in -1:1
            M = MSSTA.binaryop_mpo(
                sites, [(a,b),(c,d)], [(1,2),(1,2)]; rev_carrydirec=rev_carrydirec)

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
                xp = mod(a * x + b * y, 2^nbit)
                yp = mod(c * x + d * y, 2^nbit)
                return xp, yp
            end

            g_vec_ref = similar(g_vec)
            for x in 0:2^nbit-1, y in 0:2^nbit-1
                xp, yp = prime_xy(x, y)
                g_vec_ref[x+1, y+1] = f_vec[xp+1, yp+1]
            end
    
            #@show a, b, c, d
            @test g_vec_ref ≈ g_vec
        end

        #@test false
    end

end
