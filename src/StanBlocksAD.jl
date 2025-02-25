module StanBlocksAD
using StanBlocks, StaticArrays
import StanBlocks: square, ConstView, constview



const PB = :PB#gensym("pb")
const TMP = :TMP#gensym("tmp")
const RV = :RV#gensym("rv")
const ADJ = :ADJ

struct SReal{S,T}
    val::SVector{S,T}
end
struct Tape{M}
    mem::M
end
struct Pullback{P,T}
    pb::P
    @inline Pullback(pb, ::Type{T}) where {T} = new{typeof(pb),T}(pb)
end
abstract type AbstractDual{P,D} end
struct StrongZero end
struct WeakZero end
AbstractPlainDual{P} = AbstractDual{P,StrongZero}
struct StrippedDual{P,D} <: AbstractDual{P,D}
    deriv::D
    @inline StrippedDual(::Type{P}, deriv) where {P} = new{P,typeof(deriv)}(deriv)
end
struct Dual{P,D} <: AbstractDual{P,D}
    primal::P
    deriv::D
end
PlainDual{P} = Dual{P,StrongZero}

@inline primal(x) = x
@inline deriv(::Any) = StrongZero()
@inline primal(x::AbstractDual) = x.primal
@inline deriv(x::AbstractDual) = x.deriv
@inline primaltype(x) = typeof(primal(x))
@inline primaltype(::StrippedDual{P}) where {P} = P
@inline maybederiv(x) = x
@inline maybederiv(x::AbstractDual) = deriv(x)

@inline Base.:+(lhs::AbstractDual, rhs) = dmerge(lhs, rhs)
@inline Base.:-(lhs::AbstractDual, rhs) = dmerge(lhs, -rhs)
@inline Base.:*(::WeakZero, rhs) = WeakZero()
@inline Base.:+(lhs, ::WeakZero) = maybederiv(lhs)
@inline Base.:+(lhs::AbstractDual, ::WeakZero) = maybederiv(lhs)

@inline dmerge(lhs::AbstractDual, rhs) = dmerge(deriv(lhs), maybederiv(rhs))
@inline dmerge(::WeakZero, rhs) = maybederiv(rhs)
@inline dmerge(::StrongZero, ::Any) = StrongZero()
@inline dmerge(lhs::ConstView, rhs) = begin
    @assert lhs === maybederiv(rhs)
    lhs
end
@inline dmerge(lhs::AbstractArray, rhs) = begin
    @assert lhs === maybederiv(rhs)
    lhs
end
@inline dmerge(lhs::Real, rhs) = lhs + maybederiv(rhs)
@inline @generated dmerge(lhs::NamedTuple, rhs::NamedTuple) = begin
    @assert all(name->in(name, fieldnames(lhs)), fieldnames(rhs))
    Expr(:tuple, Expr(:parameters, [
        Expr(:kw, name, name in fieldnames(rhs) ? :(dmerge(lhs.$name, rhs.$name)) : :(lhs.$name))
        for name in fieldnames(lhs)
    ]...))
end
@inline @generated dmerge(lhs::Tuple, rhs::Tuple) = begin
    @assert fieldcount(lhs) == fieldcount(rhs)
    Expr(:tuple, [
        name in fieldnames(rhs) ? :(dmerge(lhs.$name, rhs.$name)) : :(lhs.$name)
        for name in fieldnames(lhs)
    ]...)
end

@inline Base.@propagate_inbounds dmergeindex(dx::AbstractDual, a, i) = dmergeindex(deriv(dx), a, i)
@inline Base.@propagate_inbounds dmergeindex(dx::AbstractVector, a, i) = (dx[i] = dmerge(dx[i], a); dx)
@inline Base.@propagate_inbounds dmergeindex(dx::ConstView, a, i) = (dx[i] = dmerge(dx[i], a); dx)
@inline Base.@propagate_inbounds dmergeindex(dx::Tuple, a, i) = dmergeindex(dx, a, Val(i))
@inline Base.@propagate_inbounds @generated dmergeindex(lhs::Tuple, a, ::Val{i}) where {i} = begin
    @assert i in fieldnames(lhs)
    Expr(:tuple, [
        name == i ? :(dmerge(lhs.$name, a)) : :(lhs.$name)
        for name in fieldnames(lhs)
    ]...)
end
@inline dmergeproperty(dx::AbstractDual, a, i) = dmergeproperty(deriv(dx), a, i)
@inline dmergeproperty(dx::NamedTuple, a, i) = dmergeproperty(dx, a, Val(i))
@inline @generated dmergeproperty(lhs::NamedTuple, a, ::Val{i}) where {i} = begin
    @assert i in fieldnames(lhs)
    Expr(:tuple, Expr(:parameters, [
        Expr(:kw, name, name == i ? :(dmerge(lhs.$name, a)) : :(lhs.$name))
        for name in fieldnames(lhs)
    ]...))
end


Tape() = Tape(missing)
@inline Base.getindex(t::Tape, i) = t.mem[i]
@inline Base.getindex(t::Tape{Missing}, i) = t
@inline inner_forward(pb, fargs...) = error((;MSG="MISSING DUAL OVERLOAD:", tpb=typeof(pb),tfargs=typeof(fargs)))
@inline Base.@propagate_inbounds outer_forward(pb, fargs::Vararg{Any,N}) where {N} = begin 
    dfargs = map(maybeplaindual, fargs)
    pb, rv = inner_forward(deaugment_pullback(pb), dfargs...)
    augment_pullback(pb, dfargs, rv)
end
@inline augment_pullback(pb, dfargs, rv::AbstractDual) = Pullback(pb, primaltype(rv)), rv
@inline augment_pullback(pb, dfargs, rv) = augment_pullback(pb, dfargs, Dual(rv, adj_replay(pb, dfargs...)))
@inline augment_pullback(::Missing, dfargs) = error("Missing pullback $(typeof(dfargs))")
@inline deaugment_pullback(pb) = pb
@inline adj_replay(pb, dfargs::Vararg{<:AbstractDual}) = WeakZero() 
@inline outer_pullback(pb, adj, fargs...) = begin 
    dfargs = map(maybeplaindual, fargs)
    map(redual, dfargs, inner_pullback(deaugment_pullback(pb), maybederiv(adj), dfargs...))
end
@inline inner_pullback(::Nothing, adj, fargs...) = fargs
@inline inner_pullback(pb::F, adj, fargs::Vararg{Any,N}) where {F<:Function,N} = error(N)#pb(adj, fargs...)
@inline inner_pullback(pb::F, adj, fargs::Vararg{Any,1}) where {F<:Function} = pb(adj, fargs...)
@inline inner_pullback(pb::F, adj, fargs::Vararg{Any,2}) where {F<:Function} = pb(adj, fargs...)
@inline inner_pullback(pb::F, adj, fargs::Vararg{Any,3}) where {F<:Function} = pb(adj, fargs...)
@inline inner_pullback(pb::F, adj, fargs::Vararg{Any,4}) where {F<:Function} = pb(adj, fargs...)
@inline redual(x::Dual, y) = Dual(primal(x), maybederiv(y))
@inline redual(x::StrippedDual, y) = StrippedDual(primaltype(x), maybederiv(y))
@inline plaindual(x) = Dual(x, StrongZero())
@inline maybeplaindual(x::AbstractDual) = x
@inline maybeplaindual(x) = plaindual(x)

    iscall(x, f) = Meta.isexpr(x, :call) && x.args[1] == f
    yes(arg) = true
    maps(f) = (args...)->map(f, args...)
    mapreduce2(f, op, itr) = if length(itr) == 1
        op(f(itr[1]))
    else
        mapreduce(f, op, itr)
    end 
    ereplace(e; d, descend=yes) = get(d, e, e)
    ereplace(e::Expr; d, descend=yes) = if e in keys(d)
        d[e]
    elseif descend(e)
        if e.head == :kw
            Expr(e.head, e.args[1], ereplace(e.args[2]; d, descend))
        else
            Expr(e.head, ereplace.(e.args; d, descend)...)
        end
    else
        e
    end
    xblock(x) = Meta.isexpr(x, :block) ? x : Expr(:block, x)
    xtuple(x) = Meta.isexpr(x, :tuple) ? x : Expr(:tuple, x)
    canonical_fexpr(ox::Expr) = begin 
        ass_expr = ox
        is_generated = false
        while Meta.isexpr(ass_expr, :macrocall)
            is_generated |= ass_expr.args[1] == Symbol("@generated")
            ass_expr = ass_expr.args[3]
        end
        @assert Meta.isexpr(ass_expr, :(=))
        call, rhs = ass_expr.args
        Meta.isexpr(call, :where) && (call = call.args[1])
        @assert call.head == :call dump(lhs)
        f = call.args[1]
        info = mapreduce2(maps(vcat), call.args[2:end]) do arg
            Meta.isexpr(arg, :(::)) || (arg = Expr(:(::), arg, Any))
            name, type = arg.args
            @assert isa(name, Symbol)
            (;arg, name, type)
        end
        rebuild(fargs, new_rhs) = ereplace(ox; d=Dict(
            call=>Expr(:call, fargs...),
            rhs=>xblock(new_rhs)
        ))
        merge(info, (;call, rhs, f, rebuild, is_generated))
    end
    rule_expr(ox::Expr) = begin 
        info = canonical_fexpr(ox)
        (;rhs) = info
        forward = info.f
        pb = info.arg[1]
        dinfo = mapreduce2(maps(vcat), info.arg[2:end]) do arg
            name, type = arg.args
            primal = :(primal($name))
            deriv = :(deriv($name))
            dual = :(dual($name))
            (;
                primal, 
                dual=Expr(:(::), name, :(AbstractDual{<:$type})), 
                plain=Expr(:(::), name, :(AbstractPlainDual{<:$type})),
                rep=[name=>primal,primal=>primal,deriv=>deriv,dual=>name]
            )
        end
        d = Dict(reduce(vcat, dinfo.rep))
        @assert Meta.isexpr(rhs, :block)
        if forward == :inner_forward  && length(rhs.args) <= 2
            rhs_ = rhs.args[end]
            if Meta.isexpr(rhs_, :(->)) || rhs == :missing
                rhs.args[end] = :((@inline($rhs_), ($(dinfo.primal[1])($(dinfo.primal[2:end]...)))))
            elseif rhs_ == :nothing
                rhs.args[end] = :(($rhs_, plaindual($(dinfo.primal[1])($(dinfo.primal[2:end]...)))))
            end
        end
        dual_x = info.rebuild(
            [forward, pb, dinfo.dual...], 
            ereplace(rhs; d)
        )
        plain_x = info.rebuild(
            [forward, pb, dinfo.plain...], 
            if forward == :inner_forward 
                :((nothing, plaindual($(dinfo.primal[1])($(dinfo.primal[2:end]...)))))
            elseif forward == :adj_replay
                StrongZero()
            else
                error(forward)
            end
        )
        Expr(:block, dual_x, plain_x) 
    end
nslots(info::NamedTuple) = sum(x->isa(x, Expr), info.stmts; init=0)
inline_outer(x::LineNumberNode) = x
inline_outer(x::Expr) = if x.head == :block
    Expr(:block, inline_outer.(x.args)...)
elseif x.head == :return
    x
else
    @assert x.head == :(=)
    lhs, rhs = x.args
    @assert lhs.head == :tuple
    pb, rv = lhs.args
    @assert rhs.head == :call
    outer = rhs.args[1]
    pbarg = rhs.args[2]
    fargs = rhs.args[3:end]
    @assert outer == :outer_forward
    dfargs = [:(maybeplaindual($farg)) for farg in fargs]
    quote 
        $pb, $rv = inner_forward(deaugment_pullback($pbarg), $(dfargs...))
        $pb, $rv = augment_pullback($pb, ($(dfargs...),), $rv)
    end
end
fpush!(info::NamedTuple, args...) = begin 
    i = nslots(info) + 1
    pbi, tmpi = Symbol(PB, i), Symbol(TMP, i)
    push!(info.stmts, :(($pbi, $tmpi) = outer_forward($PB[$i], $(args...))))
    tmpi
end
fw_expr_top(x::Expr; forward=:inner_forward) = begin 
    finfo = canonical_fexpr(x)
    f = finfo.f
    fargs = if forward == :inner_forward 
        [:(_::typeof($f)), finfo.arg...]
    else
        finfo.arg
    end
    fargs = mapreduce2(vcat, fargs) do arg
        name, type = arg.args
        Expr(:(::), name, :(AbstractDual{<:$type}))
    end
    finfo.rebuild(
        [forward, PB, fargs...], 
        if finfo.is_generated
            :(inline_outer(fw_expr(xblock($(finfo.rhs)))))
        else
            inline_outer(fw_expr(xblock(finfo.rhs)))
        end
    )
end
fw_expr(x::LineNumberNode; info) = push!(info.stmts, x)
fw_expr(x::Symbol; info) = x
fw_expr(x::Real; info) = (x)
fw_expr(x::Function; info) = (x)
fw_expr(x::QuoteNode; info) = x
fw_expr(x; info) = error(typeof(x))
fw_expr(x::Expr; info=(;stmts=[])) = if x.head == :block
    n1 = length(info.stmts)
    fw_expr.(x.args; info)
    Expr(:block, info.stmts[n1+1:end]...)
elseif x.head == :(=)
    @assert !Meta.isexpr(x.args[1], :call)
    n1 = length(info.stmts)
    rhs = fw_expr(x.args[2]; info)
    if length(info.stmts) == n1
        fw_expr(Expr(:call, identity, x.args[2]); info)
    end
    info.stmts[end].args[1].args[2] = x.args[1]
    x.args[1]
elseif x.head == :call
    fpush!(info, fw_expr.(x.args; info)...)
elseif x.head == :ref
    fw_expr(Expr(:call, :getindex, x.args...); info)
elseif x.head == :return
    rv = fw_expr(Expr(:(=), RV, x.args[1]); info)
    pb = :(Tape(($([Symbol(PB, i) for i in 1:nslots(info)]...),)))
    push!(info.stmts, :(return $pb, $rv))
elseif x.head == :(+=)
    lhs, rhs = x.args
    fw_expr(:($lhs = $lhs + $rhs); info)
elseif x.head == :macrocall
    fw_expr(macroexpand(@__MODULE__, x); info)
elseif x.head == :(.)
    lhs, rhs = x.args
    @assert isa(rhs, QuoteNode)
    fw_expr(:(getproperty($lhs, $rhs)); info)
else
    @warn dump(x)
    error(x)
end
pb_expr_top(x::Expr; pullback=:inner_pullback) = begin 
    locals = Dict()
    finfo = canonical_fexpr(x)
    for name in finfo.name
        locals[name] = [name]
    end
    f = finfo.f
    fargs = if pullback == :inner_pullback 
        [:(_::typeof($f)), finfo.arg...]
    else
        finfo.arg
    end
    fargs = mapreduce2(vcat, fargs) do arg
        name, type = arg.args
        Expr(:(::), name, :(AbstractDual{<:$type}))
    end
    rv = finfo.rebuild(
        [pullback, PB, ADJ, fargs...], 
        if finfo.is_generated
            :(
                Expr(:block,
                    pb_expr(fw_expr(xblock($(finfo.rhs))); info=(;locals=deepcopy($locals))),
                    Expr(:tuple, StrongZero(), $(Meta.quot.(finfo.name)...))
                )
            )
        else
            Expr(:block,
                pb_expr(fw_expr(xblock(finfo.rhs)); info=(;locals)),
                Expr(:tuple, StrongZero(), finfo.name...)
            )
        end
    )
    rv
end
pb_expr(x::Expr; info) = begin 
    @assert x.head == :block
    Expr(:block, pb_expr_fw.(x.args; info)..., pb_expr_bw.(reverse(x.args); info)...)
end
pb_expr_fw(x::LineNumberNode; info) = x
pb_expr_bw(x::LineNumberNode; info) = x
pb_expr_fw(x::Symbol; info) = get(info.locals, x, x) 
pb_expr_fw(x::Expr; info) = if x.head == :return
    nothing
else
    @assert x.head == :(=)
    lhs, rhs = x.args
    @assert lhs.head == :tuple
    _, lhs = lhs.args
    # @assert lhs ∉ keys(info.locals)
    @assert rhs.head == :call
    pb = rhs.args[2]
    fargs = rhs.args[3:end]
    lhss = get!(info.locals, lhs, Symbol[])
    push!(lhss, length(lhss) == 0 ? lhs : Symbol(lhs, "_", length(lhss)))
    :($(lhss[end]) = outer_replay($pb, $(fargs...)))
end
pb_expr_bw(x::Expr; info) = if x.head == :return
    :($RV = $ADJ)
else
    @assert x.head == :(=)
    lhs, rhs = x.args
    @assert lhs.head == :tuple
    _, lhs = lhs.args
    @assert lhs ∈ keys(info.locals)
    adj_lhs = pop!(info.locals[lhs])
    @assert rhs.head == :call
    pb = rhs.args[2]
    fargs = rhs.args[3:end]
    lhs_fargs = map(farg->last(get(info.locals, farg, [:(_)])), fargs)
    rhs_fargs = vcat(adj_lhs, map(farg->last(get(info.locals, farg, [farg])), fargs))
    :(($(lhs_fargs...),) = outer_pullback($pb, $(rhs_fargs...)))
end
auto_expr(x) = quote
    $x
    @fw $x
    @pb $x
end
overlay_expr(x) = quote
    @fw $x
    @pb $x
end
blocked(f, x::Expr) = if x.head == :block
    Expr(:block, blocked.(f, x.args)...)
else
    f(x)
end
blocked(f, x::LineNumberNode) = x
macro rule(x)
    esc(blocked(rule_expr, x))
end
macro fw(x)
    esc(blocked(fw_expr_top, x))
end
macro fw(forward, x)
    esc(fw_expr_top(x; forward))
end
macro pb(x)
    esc(blocked(pb_expr_top, x))
end
macro pb(pullback, x)
    esc(pb_expr_top(x; pullback))
end
macro auto(x)
    esc(blocked(auto_expr, x))
end
macro auto(forward, pullback, x)
    quote
        $x
        @fw $forward $x
        @pb $pullback $x
    end |> esc
end
macro overlay(x)
    esc(blocked(overlay_expr, x))
end
macro mee(x)
    esc(:(@assert false @macroexpand $x))
end

function my_sum end
@rule begin 
    @inline inner_forward(pb, f::typeof(identity), x) = (a, df, dx)->(df, dx + a)
    @inline inner_forward(pb, f::Colon, i, j) = nothing
    @inline inner_forward(pb, f::typeof(length), x) = nothing
    @inline inner_forward(pb, f::typeof(getproperty), x::Module, i::Symbol) = nothing
    @inline inner_forward(pb, f::typeof(+), x, y) = (a, df, dx, dy) -> (df, dx+a, dy+a)
    @inline inner_forward(pb, f::typeof(-), x, y) = (a, df, dx, dy) -> (df, dx+a, dy-a)
    @inline inner_forward(pb, f::typeof(-), x) = (a, df, dx) -> (df, dx-a)
    @inline inner_forward(pb, f::typeof(*), x, y) = (a, df, dx, dy) -> (df, dx+a*y, dy+a*x)
    @inline inner_forward(pb, f::typeof(/), x, y) = begin
        rv = f(x,y)
        (a, df, dx, dy) -> (df, dx+a/y, dy-a*rv/y), rv
    end
    @inline inner_forward(pb, f::typeof(log), x) = (a, df, dx) -> (df, dx + a/x)
    @inline inner_forward(pb, f::typeof(square), x) = (a, df, dx) -> (df, dx + 2 * a * x)
    @inline inner_forward(pb, f::typeof(StanBlocks.log1pexp), x) = (a, df, dx) -> (df, dx+a*StanBlocks.logistic(x))
    @inline inner_forward(pb, f::typeof(StanBlocks.logistic), x) = begin 
        rv = f(x)
        mula = rv / (1+exp(x))
        (a, df, dx) -> (df, dx+a*mula), rv
    end
    @inline inner_forward(pb, f::typeof(Base.maybeview), x, i::AbstractVector) = (a, df, dx, di) -> (df, dx, di)
    @inline inner_forward(pb, f::typeof(constview), x, i::AbstractVector) = (a, df, dx, di) -> (df, dx, di)
    @inline inner_forward(pb, f::typeof(Base.broadcasted), g, x) = (a, df, dg, dx)->(df, dmerge(dg, a.f), dmerge(dx, a.args[1]))
    @inline inner_forward(pb, f::typeof(Base.broadcasted), g, x, y) = (a, df, dg, dx, dy)->(df, dmerge(dg, a.f), dmerge(dx, a.args[1]), dmerge(dy, a.args[2]))
    @inline Base.@propagate_inbounds inner_forward(pb, f::typeof(getindex), x, i::Integer) = (a, df, dx, di) -> (df, dmergeindex(dx, a, primal(di)), di)
    @inline Base.@propagate_inbounds inner_forward(pb, f::typeof(Base.maybeview), x, i::Integer) = (a, df, dx, di) -> (df, dmergeindex(dx, a, primal(di)), di)
    @inline inner_forward(pb, f::typeof(getproperty), x, i::Symbol) = (a, df, dx, di) -> (df, dmergeproperty(dx, a, primal(di)), di)
    @inline Base.@propagate_inbounds inner_forward(pb, f::typeof(getindex), x::Union{AbstractVector,ConstView}, i::Integer) = (a, df, dx, di) -> (df, dmergeindex(dx, a, (primal(di))), di)

    @inline inner_forward(pb, f::typeof(my_sum), x::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}}) = begin 
        mem = if isa(pb, Tape{Missing})
            pb_type = Base._return_type(outer_forward, Tuple{Tape{Missing},typeof(broadcast_getindex),typeof(dual(x)),Int64})
            @assert isconcretetype(pb_type) outer_forward(Tape(), broadcast_getindex, dual(x), 1)
            pb_type = pb_type.types[1]
            Vector{pb_type}(undef, length(x))
        else
            resize!(pb.mem, length(x))
        end
        rv = 0.
        (mem[1], drv) = outer_forward(Tape(), broadcast_getindex, dual(x), 1)
        rv = primal(drv)
        @inbounds @simd for i in eachindex(x)[2:end]
            (mem[i], drv) = outer_forward(Tape(), broadcast_getindex, dual(x), i)
            rv += primal(drv)
        end
        sum_pb(a, df, dx) = begin 
            (_, dx, _) = outer_pullback(mem[1], a, broadcast_getindex, dx, 1)
            @inbounds @simd for i in eachindex(mem)[2:end]
                (_, dx, _) = outer_pullback(mem[i], a, broadcast_getindex, dx, i) 
            end
            (df, dx)
        end
        sum_pb, (rv)
    end

    @inline adj_replay(pb, f::typeof(getproperty), x, i::Symbol) = (f(deriv(x), (i)))
    @inline Base.@propagate_inbounds adj_replay(pb, f::typeof(getindex), x, i::Integer) = (f(deriv(x), (i)))
    @inline Base.@propagate_inbounds adj_replay(pb, f::typeof(Base.maybeview), x, i) = (f(deriv(x), (i)))
    @inline Base.@propagate_inbounds adj_replay(pb, f::typeof(constview), x, i::AbstractVector) = (f(deriv(x), (i)))
    @inline adj_replay(pb, f::typeof(Base.broadcasted), g, x) = ((;f=deriv(g), args=(deriv(x),)))
    @inline adj_replay(pb, f::typeof(Base.broadcasted), g, x, y) = ((;f=deriv(g), args=(deriv(x), deriv(y))))
end


broadcast_getindex_expr(::Type{T}, x=:x) where {T} = :(broadcast_getindex($x, i))
broadcast_getindex_expr(::Type{Base.Broadcast.Broadcasted{Style,Axes,F,Args}}, x=:x) where {Style<:Base.Broadcast.DefaultArrayStyle,Axes,F,Args<:Tuple} = begin 
    subs = map(fieldtypes(Args), 1:fieldcount(Args)) do Ti, i
        broadcast_getindex_expr(Ti, :($x.args[$i]))
    end
    x == :x ? :(return $x.f($(subs...))) : :($x.f($(subs...)))
end
@auto begin 
    @inline Base.@propagate_inbounds @generated broadcast_getindex(x::Base.Broadcast.Broadcasted{Style,Axes,F,Args}, i) where {Style<:Base.Broadcast.DefaultArrayStyle,Axes,F,Args<:Tuple} = broadcast_getindex_expr(Base.Broadcast.Broadcasted{Style,Axes,F,Args})
    @inline Base.@propagate_inbounds broadcast_getindex(x::AbstractVector, i::Int64) = return x[i]
    @inline Base.@propagate_inbounds broadcast_getindex(x::ConstView, i::Int64) = return x[i]
    @inline Base.@propagate_inbounds broadcast_getindex(x::Real, i::Int64) = return x
    @inline my_sum(x::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}) = return broadcast_getindex(x,1)
    @inline Base.@propagate_inbounds broadcast_getindex(x::SReal, i::Int64) = return x
    StanBlocks.normal_lpdf(y, loc, scale::SReal) = begin
        s2 = StanBlocks.@broadcasted square(y-loc)
        return -(log(scale) * length(s2)+.5*my_sum(s2)/square(scale))
    end
end
@inline Base.sum(x::SReal) = sum(x.val)
@overlay begin 
    @inline Base.sum(x::Float64) = return identity(x)
    @inline Base.sum(x::SReal) = return identity(x)
    @inline Base.sum(x::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}) = return broadcast_getindex(x,1)
end
my_sum(x::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{1}}) = begin 
    rv = broadcast_getindex(x, 1)
    @inbounds @simd for i in eachindex(x)[2:end]
        rv += broadcast_getindex(x, i)
    end
    rv
end

@inline maybeweak(x::Real) = WeakZero()
@inline maybeweak(x::StrongZero) = x
@inline maybeweak(x::WeakZero) = x
@inline maybeweak(x::Tuple) = map(maybeweak, x)
@inline maybeweak(x::NamedTuple) = map(maybeweak, x)
@inline maybeweak(x::AbstractArray) = x
@inline maybeweak(x::ConstView) = x
@inline maybeweak(x::SReal) = WeakZero()
@inline rvtype(::Pullback{P,T}) where {P,T} = T
@inline deaugment_pullback(pb::Pullback) = pb.pb
@inline Base.@propagate_inbounds inner_replay(pb, dfargs::Vararg{<:PlainDual}) = inner_forward(deaugment_pullback(pb), dfargs...)[2]
@inline inner_replay(pb, dfargs::Vararg{<:AbstractPlainDual}) = StrippedDual(rvtype(pb), StrongZero())
@inline Base.@propagate_inbounds inner_replay(pb, dfargs::Vararg{<:AbstractDual}) = StrippedDual(
    rvtype(pb), maybeweak(adj_replay(pb, dfargs...))
)
@inline Base.@propagate_inbounds outer_replay(pb::Pullback, fargs...) = begin 
    dfargs = map(maybeplaindual, fargs)
    inner_replay(pb, dfargs...)::AbstractDual
end


@inline Base.zero(::Type{SReal{S,T}}) where {S,T} = SReal(zero(SVector{S,T}))
@inline @generated Base.convert(::Type{SReal{S,T}}, x::Real) where {S,T} = :(
    SReal(SVector{S,T}($(fill(:x, S)...)))
)
@inline Base.:+(x::T, y::T) where {T<:SReal} = SReal(x.val .+ y.val)
@inline Base.:-(x::T, y::T) where {T<:SReal} = SReal(x.val .- y.val)
@inline Base.:-(x::SReal) = SReal(.-x.val)
@inline Base.:*(x::T, y::T) where {T<:SReal} = SReal(x.val .* y.val)
@inline Base.:/(x::T, y::T) where {T<:SReal} = SReal(x.val ./ y.val)
@inline StanBlocks.square(x::SReal) = SReal(StanBlocks.square.(x.val))
@inline Base.exp(x::SReal) = SReal(Base.exp.(x.val))
@inline Base.log(x::SReal) = SReal(Base.log.(x.val))
@inline StanBlocks.log1pexp(x::SReal) = SReal(StanBlocks.log1pexp.(x.val))
@inline StanBlocks.logistic(x::SReal) = SReal(StanBlocks.logistic.(x.val))
@inline @generated Base.randn(rng, ::Type{SReal{S,T}}) where {S,T} = :(
    SReal(SVector{S,T}($(fill(:(randn(rng)), S)...)))
)

@inline Base.broadcastable(x::SReal) = x
@inline Base.Broadcast.BroadcastStyle(::Type{<:SReal}) = Base.Broadcast.DefaultArrayStyle{0}()
@inline Base.length(x::SReal) = 1
@inline Base.size(x::SReal) = (1,)
@inline Base.:+(x::Real, y::SReal) = SReal(x .+ y.val)
@inline Base.:-(x::Real, y::SReal) = SReal(x .- y.val)
@inline Base.:-(x::SReal, y::Real) = SReal(x.val .- y)
@inline Base.:*(x::Real, y::SReal) = SReal(x .* y.val)
@inline Base.:*(x::SReal, y::Real) = SReal(x.val .* y)
@inline Base.:/(x::Real, y::SReal) = SReal(x ./ y.val)
@inline Base.:/(x::SReal, y::Real) = SReal(x.val ./ y)
@inline dmerge(lhs::SReal, rhs::SReal) = SReal(dmerge.(lhs.val, rhs.val))
@inline dmerge(lhs::SReal, ::WeakZero) = lhs
@inline dmerge(lhs::SReal, rhs) = SReal(dmerge.(lhs.val, rhs))

end # module StanBlocksAD
