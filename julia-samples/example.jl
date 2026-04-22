module Example

function add(x,y)
    return x+y
end

function greet(name)
    println("Hello, ",name,"!")
end

struct Point
    x::Float64
    y::Float64
end

function distance(p1::Point,p2::Point)
    sqrt((p1.x-p2.x)^2+(p1.y-p2.y)^2)
end

function log_result(x)
    @info "result" value=x
end

end
