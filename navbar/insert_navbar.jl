using HTTP

function read_file(filename::String)
    open(filename, "r") do io
        read(io, String)
    end
end

function write_file(filename::String, contents::String)
    open(filename, "w") do io
        write(io, contents)
    end
end

function should_exclude(filename::String, patterns::Vector{String})
    for pat in patterns
        if occursin(pat, filename)
            return true
        end
    end
    return false
end

function remove_existing_navbar(html::String)
    start_marker = "<!-- NAVBAR START -->"
    end_marker   = "<!-- NAVBAR END -->"
    while occursin(start_marker, html) && occursin(end_marker, html)
        start_idx_range = findfirst(start_marker, html)
        end_idx_range   = findfirst(end_marker, html)
        start_idx = first(start_idx_range)
        end_idx = first(end_idx_range)
        prefix = start_idx > 1 ? html[1:start_idx-1] : ""
        suffix = lstrip(html[end_idx + length(end_marker) : end])
        html = string(prefix, suffix)
    end
    return html
end

function wrap_navbar(navbar_html::String)
    if !occursin("NAVBAR START", navbar_html) || !occursin("NAVBAR END", navbar_html)
        return "<!-- NAVBAR START -->\n" * navbar_html * "\n<!-- NAVBAR END -->"
    else
        return navbar_html
    end
end

function insert_navbar(html::String, navbar_html::String)
    html = remove_existing_navbar(html)
    m = match(r"(?i)(<body[^>]*>)", html)
    if m === nothing
        error("Could not find <body> tag in the HTML.")
    end
    prefix = m.match
    inserted = string(prefix, "\n", navbar_html, "\n")
    html = replace(html, prefix => inserted; count = 1)
    return html
end

function process_file(filename::String, navbar_html::String)
    println("Processing: $filename")
    html = read_file(filename)
    html = insert_navbar(html, navbar_html)
    write_file(filename, html)
    println("Updated: $filename")
end

function main()
    if length(ARGS) < 2
        println("Usage: julia update_navbar.jl <html-file-or-directory> <navbar-file-or-url> [--exclude \"pat1,pat2,...\"]")
        return
    end
    target = ARGS[1]
    navbar_source = ARGS[2]
    
    exclude_patterns = String[]
    if length(ARGS) ≥ 4 && ARGS[3] == "--exclude"
        exclude_patterns = map(x -> string(strip(x)), split(ARGS[4], ','))
    end

    navbar_html = ""
    if startswith(lowercase(navbar_source), "http")
        resp = HTTP.get(navbar_source)
        if resp.status != 200
            error("Failed to download navbar from $navbar_source")
        end
        navbar_html = String(resp.body)
    else
        navbar_html = read_file(navbar_source)
    end
    navbar_html = string(navbar_html)
    navbar_html = wrap_navbar(navbar_html)

    if isfile(target)
        if !should_exclude(target, exclude_patterns)
            process_file(target, navbar_html)
        else
            println("Skipping excluded file: $target")
        end
    elseif isdir(target)
        for (root, _, files) in walkdir(target)
            for file in files
                if endswith(file, ".html")
                    fullpath = joinpath(root, file)
                    if !should_exclude(fullpath, exclude_patterns)
                        process_file(fullpath, navbar_html)
                    else
                        println("Skipping excluded file: $fullpath")
                    end
                end
            end
        end
    else
        error("Target $target is neither a file nor a directory.")
    end
end

main()
