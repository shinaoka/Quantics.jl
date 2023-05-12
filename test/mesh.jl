using Test
import MSSTA

@testset "mesh.jl" begin
    @testset "_preprocess_matmul" begin
        m = MSSTA.DiscreteMesh{3}(5)

        for idx in [(1, 1, 1), (1, 1, 2)]
            c = MSSTA.originalcoordinate(m, idx)
            @test MSSTA.meshindex(m, c) == idx
        end
    end
end