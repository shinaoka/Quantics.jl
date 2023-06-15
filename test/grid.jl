using Test
import MSSTA

@testset "grid.jl" begin
    @testset "DiscreteGrid" begin
        m = MSSTA.InherentDiscreteGrid{3}(5)

        for idx in [(1, 1, 1), (1, 1, 2)]
            c = MSSTA.grididx_to_origcoord(m, idx)
            @test MSSTA.origcoord_to_grididx(m, c) == idx
        end
    end

    @testset "DiscretizedGrid" begin
        @testset "1D" begin
            R = 5
            grid_min = 0.1
            grid_max = 2.0
            dx = (grid_max - grid_min) / 2^R
            g = MSSTA.DiscretizedGrid{1}(R, (grid_min,), (grid_max,))
            @test @inferred(MSSTA.origcoord_to_grididx(g, 0.999999 * dx + grid_min)) == 1
            @test MSSTA.origcoord_to_grididx(g, 1.999999 * dx + grid_min) == 2
            @test MSSTA.origcoord_to_grididx(g, grid_max - 1e-9 * dx) == 2^R
        end

        @testset "2D" begin
            R = 5
            d = 2
            grid_min = (0.1, 0.1)
            grid_max = (2.0, 2.0)
            dx = (grid_max .- grid_min) ./ 2^R
            g = MSSTA.DiscretizedGrid{d}(R, grid_min, grid_max)

            cs = [
                0.999999 .* dx .+ grid_min,
                1.999999 .* dx .+ grid_min,
                grid_max .- 1e-9 .* dx
            ]
            refs = [1, 2, 2^R]

            for (c, ref) in zip(cs, refs)
                @inferred(MSSTA.origcoord_to_grididx(g, c))
                @test all(MSSTA.origcoord_to_grididx(g, c) .== ref)
            end
        end
    end
end

nothing
