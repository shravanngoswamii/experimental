module Example

function add(x,y)
    return x+y
end

function distance(ax,ay,bx,by)
    sqrt((ax-bx)^2+(ay-by)^2)
end

function clamp_val(x,lo,hi)
    x<lo ? lo : x>hi ? hi : x
end

end
