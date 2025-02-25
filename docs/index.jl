
using StanBlocks, Chairmarks, DataFrames
import PosteriorDB, Enzyme, Mooncake, DifferentiationInterface
import StanBlocksAD: @auto, @fw, @pb, AbstractDual, Dual, Tape, primal, deriv, SReal, constview, augment_pullback, deaugment_pullback, maybeplaindual, inner_forward, outer_replay, outer_pullback

pdb = PosteriorDB.database()
post = PosteriorDB.posterior(pdb, "radon_mod-radon_county")
jlpdf = StanBlocks.julia_implementation(post)
(;J, county, y) = jlpdf.f


n = StanBlocks.dimension(jlpdf)
radon_county_lpdf(;J, county, y, BATCH_TYPE=Float64) = begin
    @auto lpdf_fw lpdf_pb lpdf(x) = @views begin
        target = 0.
        a = x[1:J]
        mu_a = x[J+1]
        tmp = x[J+2]
        target += log(100)-tmp-2*StanBlocks.log1pexp(-tmp)
        sigma_a = 100 * StanBlocks.logistic(tmp)
        tmp = x[J+3]
        target += log(100)-tmp-2*StanBlocks.log1pexp(-tmp)
        sigma_y = 100 * StanBlocks.logistic(tmp)
        # y_hat = x[county]
        y_hat = constview(x, county)
        target += StanBlocks.normal_lpdf(mu_a, 0, 1)
        target += StanBlocks.normal_lpdf(a, mu_a, sigma_a)
        target += StanBlocks.normal_lpdf(y, y_hat, sigma_y)
        return sum(target)
    end
    xg = Dual(zeros(BATCH_TYPE, J+3), zeros(BATCH_TYPE, J+3))
    pb, _ = lpdf_fw(Tape(), xg)
    lpdfg(x) = begin 
        primal(xg) .= x
        deriv(xg) .= 0
        pb_, rv = lpdf_fw(pb, xg)
        lpdf_pb(pb_, Dual(primal(rv), 1.), xg)
        primal(rv), deriv(xg)
    end
    StanBlocks.VectorPosterior(lpdf, missing, missing, n), lpdfg
end

@inline infoandpass(arg) = (@info arg; arg)
round2(;kwargs...) = round2((;kwargs...))
round2(x::Float64) = round(x; sigdigits=2)
round2(x::Integer) = x
round2(x::NamedTuple) = map(round2, x)
round2(x::Tuple) = map(round2, x)
areapprox(args...) = begin
    uargs = map(untangent, args)
    for i in 1:length(args), j in 1:i-1
        isapprox(uargs[i], uargs[j]) || return false
    end 
    return true
end
untangent(x::Real) = x
untangent(x::SReal) = sum(x.val)
untangent(x::Vector{<:Real}) = x
untangent(x::Vector{<:SReal}) = mapreduce(xi->[xi.val...], hcat, x)
untangent(x::Vector{<:Mooncake.Tangent}) = mapreduce(xi->[xi.fields.val.fields.data...], hcat, x)
begin
BATCH_TYPES = (Float64, SReal{1,Float64}, SReal{2,Float64}, SReal{4,Float64}, SReal{8,Float64}, SReal{16,Float64})
timings = mapreduce(hcat, BATCH_TYPES) do BATCH_TYPE
    display(BATCH_TYPE)
    @time mlpdf, mlpdfg = radon_county_lpdf(;J, county, y, BATCH_TYPE)
    @time mlpdfg2 = StanBlocks.with_gradient(mlpdf, DifferentiationInterface.AutoMooncake(;config=nothing); T=BATCH_TYPE).g
    @time mlpdfg3 = StanBlocks.with_gradient(mlpdf, DifferentiationInterface.AutoEnzyme(;mode=Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal), function_annotation=Enzyme.Const); T=BATCH_TYPE).g
    x = randn(BATCH_TYPE, n)
    @assert areapprox(mlpdf(x), mlpdfg(x)[1], mlpdfg2(x)[1], mlpdfg3(x)[1])
    @assert areapprox(mlpdfg(x)[2], mlpdfg2(x)[2], mlpdfg3(x)[2])
    rv = [(@be randn(BATCH_TYPE, n) mlpdf), (@be randn(BATCH_TYPE, n) mlpdfg), (@be randn(BATCH_TYPE, n) mlpdfg2), (@be randn(BATCH_TYPE, n) mlpdfg3)]
    map(rv) do rvi
        minimum(rvi).time * 1e6
    end
end
end
begin

works = [1 1 2 4 8 16]
columns = Symbol.("Batch=", vec(works)) 
columns[1] = Symbol("Float64")
speedups = DataFrame(round2.((works ./ timings) .* minimum(timings[2:end, 1])), columns)
insertcols!(speedups, 1, :AD=>["Plain", "Mine", "Mooncake", "Enzyme"])
end