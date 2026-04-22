module Utils

function read_csv(path; sep=',', header=true)
    open(path, "r") do io
        lines = readlines(io)
        if header
            lines = lines[2:end]
        end
        [split(l, sep) for l in lines]
    end
end

function retry(f, n=3; delay=1.0)
    for i = 1:n
        try
            return f()
        catch e
            i==n && rethrow(e)
            sleep(delay)
        end
    end
end

macro timed_log(expr)
    quote
        t = @elapsed result=$(esc(expr))
        @info "elapsed" seconds=t
        result
    end
end

end
