using MultiScaleSpaceTimes
using Documenter

DocMeta.setdocmeta!(MultiScaleSpaceTimes, :DocTestSetup, :(using MultiScaleSpaceTimes); recursive=true)

makedocs(;
    modules=[MultiScaleSpaceTimes],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    repo="https://github.com/shinaoka/MultiScaleSpaceTimes.jl/blob/{commit}{path}#{line}",
    sitename="MultiScaleSpaceTimes.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
