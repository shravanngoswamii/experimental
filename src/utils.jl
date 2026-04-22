module Utils

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

function log_event(msg; level=:info)
    if level==:info
        @info msg
    elseif level==:warn
        @warn msg
    else
        @error msg
    end
end

end
