module Stats

function mean(xs)
    sum(xs)/length(xs)
end

function variance(xs)
    m = mean(xs)
    sum((x-m)^2 for x in xs)/(length(xs)-1)
end

function normalize(xs)
    m = mean(xs)
    s = sqrt(variance(xs))
    [(x-m)/s for x in xs]
end

function clamp_val(x,lo,hi)
    if x<lo
        lo
    elseif x>hi
        hi
    else
        x
    end
end

end
