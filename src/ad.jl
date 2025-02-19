function optim_only_fg!(func, backend::ADTypes.AbstractADType)
    Optim.only_fg!() do F, G, z
        if G != nothing
            f, g = DI.value_and_gradient(func, backend, z)
            G .= first(g)
            return f
        end
        if F != nothing
            return func(z)
        end
    end
end
