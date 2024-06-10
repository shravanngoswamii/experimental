#!/bin/bash

# URL of the navbar_config.yml file
CONFIG_URL="https://raw.githubusercontent.com/shravanngoswamii/experimental/main/test/navbar_config.yml"

# Download the navbar_config.yml file
curl -sL $CONFIG_URL -o navbar_config.yml

# Function to generate HTML for navigation bar
generate_navbar_html() {
    local yaml_file=$1
    local navbar_html=""

    # CSS Styles
    local css=$(cat <<EOF
<!-- Insert CSS styles here -->
<style>
    .dropdown-content {
        display: none;
        position: absolute;
        margin-top: 43px;
        /* margin: auto; */
        /* left: 50%; */
        transform: translateX(-25%);
        z-index: 1;
        border: #8faad2 solid 1px;
        }
        
        .dropdown.active .dropdown-content {
        background-color: #0e5964;
        display: flex;
        width: fit-content;
    }

    .dropdown-content .nav-link {
        padding: 10px 15px;
        white-space: nowrap;
    }

    /* Sub-dropdown styles */
    .sub-dropdown .sub-dropdown-content {
        display: none;
        position: absolute;
        top: 100%;
        /* Position below the parent li */
        background-color: #0e5964;
        box-shadow: 0px 8px 16px 0px rgba(0, 0, 0, 0.2);
        z-index: 2;
        text-align: left;
        width: fit-content;
    }

    .sub-dropdown.active .sub-dropdown-content {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    html {
        --navbar-height: 55px;
        scroll-padding-top: calc(var(--navbar-height) + 1rem);
    }

    .navigation {
        position: fixed;
        height: 60px;
        top: 0;
        width: 100%;
        background-color: #073c44;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        padding: 0 17px;
        transition: top 0.3s;
    }

    .navbar-logo {
        margin-left: 10px;
    }

    .nav-links {
        display: flex;
        align-items: center;
        list-style-type: none;
        margin: 0;
        padding: 0;
        flex-grow: 1;
    }

    .nav-links li {
        margin-left: 16px !important;
    }

    .nav-link {
        color: white !important;
        text-decoration: none;
        font-size: 17px !important;
    }

    .nav-link:hover {
        color: #8faad2 !important;
    }
    /* .dropdown-toggle {
        color: white !important;
        text-decoration: none;
        font-size: 17px !important;
        cursor: pointer;
    } */

    .menu-toggle {
        display: none;
        font-size: 24px;
        color: white;
        cursor: pointer;
    }

    /* Documenter css tweaks */
    .docs-sidebar {
        margin-top: 60px;
        /* padding-top: calc(var(--navbar-height) + -3rem) !important; */
    }

    #documenter {
        margin-top: 60px;
    }

    /* Responsive styling */
    @media (max-width: 768px) {
        .dropdown {
            flex-direction: column;
        }

        .dropdown-content {
            margin: auto;
            flex-direction: column;
            position: relative;
            transform: none;
        }
      

        .sub-dropdown .sub-dropdown-content {
            position: relative;
            top: 0;
        }

        .nav-links {
            display: none;
            flex-direction: column;
            width: 100%;
            background-color: #073c44;
            position: absolute;
            top: 60px;
            left: 0;
            padding: 10px 0;
        }

        .nav-links.show {
            display: flex;
        }

        .nav-links li {
            margin: 10px 0;
            text-align: center;
        }

        .menu-toggle {
            display: block;
            margin-left: auto;
        }

        .navigation.hide {
            top: -60px;
        }
    }
</style>
EOF
    )

    # JavaScript
    local js=$(cat <<EOF
<!-- Insert JavaScript here -->
<script>
    document.addEventListener('DOMContentLoaded', function () {
        const dropdownToggle = document.querySelector('.dropdown-toggle');
        const dropdown = document.querySelector('.dropdown');
        const menuToggle = document.querySelector('.menu-toggle');
        const navLinks = document.querySelector('.nav-links');
        const nav = document.querySelector('.navigation');
        const subDropdownToggles = document.querySelectorAll('.sub-dropdown-toggle');
        let lastScrollY = window.scrollY;

        // Toggle dropdown menu
        dropdownToggle.addEventListener('click', event => {
            event.preventDefault();
            dropdown.classList.toggle('active');
        });

        // Toggle sub-dropdowns and ensure only one is open at a time
        subDropdownToggles.forEach(toggle => {
            toggle.addEventListener('click', event => {
                event.preventDefault();
                const subDropdown = toggle.parentElement;
                document.querySelectorAll('.sub-dropdown').forEach(item => {
                    if (item !== subDropdown) item.classList.remove('active');
                });
                subDropdown.classList.toggle('active');
            });
        });

        // Toggle main menu for mobile
        menuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('show');
        });

        // Close menus if clicked outside
        document.addEventListener('click', event => {
            if (!dropdown.contains(event.target) && !dropdownToggle.contains(event.target)) {
                dropdown.classList.remove('active');
            }
            if (!navLinks.contains(event.target) && !menuToggle.contains(event.target)) {
                navLinks.classList.remove('show');
            }
        });

        // Hide navigation bar on scroll down in mobile view
        window.addEventListener('scroll', () => {
            if (window.innerWidth <= 768) {
                nav.classList.toggle('hide', window.scrollY > lastScrollY);
                lastScrollY = window.scrollY;
            }
        });
    });

</script>
EOF
    )

    # Read brand information
    local brand_name=$(awk '/brand:/{flag=1;next}/link:/{flag=0}flag' $yaml_file | grep "name" | awk '{print $2}')
    local brand_link=$(awk '/brand:/{flag=1;next}/logo:/{flag=0}flag' $yaml_file | grep "link" | awk '{print $2}')
    local brand_logo=$(awk '/brand:/{flag=1;next}/links:/{flag=0}flag' $yaml_file | grep "logo" | awk '{print $2}')

    # Add brand HTML
    navbar_html+="<nav class=\"navigation\">"
    navbar_html+="<a href=\"$brand_link\"><img src=\"$brand_logo\" alt=\"$brand_name Logo\" class=\"navbar-logo\" height=\"24px\" width=\"40px\"></a>"
    navbar_html+="<a style=\"color: white !important; font-size: 21.25px !important; margin-left: 10px;\" href=\"$brand_link\">$brand_name</a>"

    # Start nav links
    navbar_html+="<ul class=\"nav-links\">"

    local indent=0

    # Function to process sublinks
    process_sublinks() {
        local sublinks="$1"
        local indent_level=$2

        while read subline; do
            if echo "$subline" | grep -q "name:"; then
                sublink_name=$(echo "$subline" | awk '{print $2}')
            fi
            if echo "$subline" | grep -q "link:"; then
                sublink_url=$(echo "$subline" | awk '{print $2}')
                navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))<li><a class=\"nav-link\" href=\"$sublink_url\">$sublink_name</a></li>"
            fi
            if echo "$subline" | grep -q "dropdown:"; then
                navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))<li class=\"sub-dropdown\">"
                navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))<a class=\"nav-link sub-dropdown-toggle\" href=\"#\">$sublink_name▾</a>"
                navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))<ul class=\"sub-dropdown-content\">"
            fi
            if echo "$subline" | grep -q "sublinks:"; then
                process_sublinks "$sublinks" $((indent_level + 2))
            fi
        done <<< "$(awk -v level=$((indent_level / 2)) '/^ {0,2}/ {if (NR==1) next; indent=length($0)-length(ltrim($0)); if (indent<=level) exit; print $0}' <<< "$sublinks")"

        navbar_html+="$(printf ' %.0s' $(seq 1 $((indent_level - 2))))</ul>"
        navbar_html+="$(printf ' %.0s' $(seq 1 $((indent_level - 2))))</li>"
    }

    # Read navigation links
    while read line; do
        if echo "$line" | grep -q "name:"; then
            link_name=$(echo "$line" | awk '{print $2}')
        fi
        if echo "$line" | grep -q "link:"; then
            link_url=$(echo "$line" | awk '{print $2}')
            navbar_html+="<li><a class=\"nav-link\" href=\"$link_url\">$link_name</a></li>"
        fi
        if echo "$line" | grep -q "dropdown:"; then
            navbar_html+="<li class=\"dropdown\">"
            navbar_html+="<a class=\"nav-link dropdown-toggle\" href=\"#\">$link_name▾</a>"
            navbar_html+="<ul class=\"dropdown-content\">"
        fi
        if echo "$line" | grep -q "sublinks:"; then
            sublinks=$(awk '/sublinks:/{flag=1;next}/^$/flag' <<< "$line")
            process_sublinks "$sublinks" $((indent + 2))
        fi
    done < <(awk '/links:/{flag=1;next}/^$/{flag=0}flag' $yaml_file)

    # Close nav links and nav
    navbar_html+="</ul>"
    navbar_html+="<span class=\"menu-toggle\">&#9776;</span>"
    navbar_html+="</nav>"

    # Append CSS and JavaScript
    navbar_html="$css\n$navbar_html\n$js"

    echo "$navbar_html"
}

# Generate the navbar HTML from YAML config and save to navbar.html
generate_navbar_html "navbar_config.yml" > navbar.html

echo "Generated navbar.html successfully!"
