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
    local brand_name=$(awk -F ': ' '/brand:/{flag=1;next}/links:/{flag=0}flag' $yaml_file | grep "name" | awk -F ': ' '{print $2}' | tr -d '"')
    local brand_link=$(awk -F ': ' '/brand:/{flag=1;next}/logo:/{flag=0}flag' $yaml_file | grep "link" | awk -F ': ' '{print $2}' | tr -d '"')
    local brand_logo=$(awk -F ': ' '/brand:/{flag=1;next}/links:/{flag=0}flag' $yaml_file | grep "logo" | awk -F ': ' '{print $2}' | tr -d '"')

    # Add brand HTML
    navbar_html+="<nav class=\"navigation\">"
    navbar_html+="<a href=\"$brand_link\"><img src=\"$brand_logo\" alt=\"$brand_name Logo\" class=\"navbar-logo\" height=\"24px\" width=\"40px\"></a>"
    navbar_html+="<a style=\"color: white !important; font-size: 21.25px !important; margin-left: 10px;\" href=\"$brand_link\">$brand_name</a>"

    # Start nav links
    navbar_html+="<ul class=\"nav-links\">"

    # Function to process links and sublinks
    process_links() {
        local yaml_data="$1"
        local indent_level=$2

        while read -r line; do
            local name=$(echo "$line" | awk -F ': ' '/name/ {print $2}' | tr -d '"')
            local link=$(echo "$line" | awk -F ': ' '/link/ {print $2}' | tr -d '"')
            local dropdown=$(echo "$line" | awk -F ': ' '/dropdown/ {print $2}' | tr -d '"')

            if [ "$dropdown" == "true" ]; then
                navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))<li class=\"dropdown\">"
                navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))<a class=\"nav-link dropdown-toggle\" href=\"#\">$name▾</a>"
                navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))<ul class=\"dropdown-content\">"
                local sublinks=$(awk -v start="sublinks:" '/dropdown: true/{flag=1;next}/- name:/{flag=0}flag' <<< "$line")
                process_links "$sublinks" $((indent_level + 2))
                navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))</ul>"
                navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))</li>"
            else
                if [ -n "$name" ] && [ -n "$link" ]; then
                    navbar_html+="$(printf ' %.0s' $(seq 1 $indent_level))<li><a class=\"nav-link\" href=\"$link\">$name</a></li>"
                fi
            fi
        done <<< "$yaml_data"
    }

    # Read navigation links
    links=$(awk '/links:/{flag=1;next}/^$/{flag=0}flag' $yaml_file)
    process_links "$links" 2

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
