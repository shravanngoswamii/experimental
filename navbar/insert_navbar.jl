using Gumbo, HTTP, URIs, Logging

function process_file!(html_path, navbar_content, excluded_paths)
    any(excluded_paths) do path
        occursin(path, html_path)
    end && return @info "Skipping excluded file: $html_path"

    @info "Processing: $html_path"
    
    # Read and parse HTML
    html_str = read(html_path, String)
    doc = parsehtml(html_str)
    
    # Remove existing navbar
    body = doc.root[2]  # Assuming standard structure: html > head + body
    nav_comments = []
    
    # Find NAVBAR comments
    for elem in PreOrderDFS(body)
        isa(elem, HTMLComment) || continue
        occursin("NAVBAR START", elem.text) && push!(nav_comments, elem)
        occursin("NAVBAR END", elem.text) && push!(nav_comments, elem)
    end
    
    # Remove elements between comments
    if length(nav_comments) == 2
        start_idx = findfirst(==(nav_comments[1]), body.children)
        end_idx = findfirst(==(nav_comments[2]), body.children)
        
        if start_idx !== nothing && end_idx !== nothing && start_idx < end_idx
            deleteat!(body.children, start_idx:end_idx)
        end
    end
    
    # Insert new navbar after <body> tag
    navbar_node = parsehtml(navbar_content).root[2].children[1]  # Extract body content
    insert!(body.children, 1, navbar_node)
    
    # Write modified HTML back
    write(html_path, string(doc))
end

function main()
    # Parse command-line arguments
    if length(ARGS) < 2
        println("Usage: julia insert_navbar.jl <html-directory> <navbar-url> [--exclude path1,path2,...]")
        exit(1)
    end

    html_dir = ARGS[1]
    navbar_source = ARGS[2]
    exclude_paths = []
    
    # Parse exclude paths
    if "--exclude" in ARGS
        idx = findfirst(==("--exclude"), ARGS)
        exclude_paths = split(ARGS[idx+1], ',')
    end

    # Get navbar content
    navbar_content = if occursin(r"^https?://", navbar_source)
        HTTP.get(navbar_source).body |> String
    else
        read(navbar_source, String)
    end

    # Process files
    for (root, dirs, files) in walkdir(html_dir)
        any(exclude_paths) do path
            occursin(path, root)
        end && continue
        
        for file in files
            endswith(file, ".html") || continue
            html_path = joinpath(root, file)
            try
                process_file!(html_path, navbar_content, exclude_paths)
            catch e
                @error "Failed to process $html_path" exception=(e, catch_backtrace())
            end
        end
    end
end

main()