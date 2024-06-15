---
layout: null
excluded_in_search: true
---
$(document).ready(function(){
    // Mapping of versions to linked version numbers
    var versionLinks = {
        "v0.31": "v0.31.4",
        "v0.30": "v0.30.5",
        "v0.29": "v0.29.3",
        "v0.28": "v0.28.8",
        "v0.27": "v0.27.6",
        "v0.26": "v0.26.2",
        "v0.25": "v0.25.1",
        "v0.24": "v0.24.7"
    };

    var url_parts = /(.*:\/\/[^/]+\/)(.+?)(\/.*)/.exec(window.location.href);

    $(".dropdown > a").text(url_parts[2]);

    if(url_parts[3].length > 1 && url_parts[3][1] != "#") {
        $(".dropdown").click(function(evt){
            $(".dropdown > div.dropdown-menu").toggleClass("show");
            return false;
        });

        $('body').click(function(evt){
            if($(evt.target).closest('div.dropdown-menu').length) {
                return true;
            }
            if($(evt.target).closest('div.dropdown').length) {
                return true;
            }
            $(".dropdown > div.dropdown-menu").removeClass("show");
        });
    }

    var current_ver = url_parts[2].replace(/\.0+$/, '');

    $.each(DOC_VERSIONS, function(index, value) {
        var linkedVersion = versionLinks[value];
        if (!linkedVersion) return;  // Skip if there's no linked version

        if(value == current_ver) {
            // mobile
            $("select#version-selector").append(
                $('<option value="' + linkedVersion + '" selected="selected">' + value + '</option>'));
            return;
        }
        // desktop
        $(".dropdown > div.dropdown-menu").append(
            $('<a class="dropdown-item" href="#" data-linked-version="' + linkedVersion + '">' + value + '</a>'));
        // mobile
        $("select#version-selector").append(
            $('<option value="' + linkedVersion + '">' + value + '</option>'));
    });

    $(".dropdown > div.dropdown-menu > a").on("click", function() {
        var linkedVersion = $(this).data('linked-version');
        if (!linkedVersion || linkedVersion == url_parts[2]) return;
        var new_url = window.location.href.replace(/\/ui\/[^\/]+/, "/ui/" + linkedVersion);
        window.location.href = new_url;
    });

    $("select#version-selector").change(function() {
        var linkedVersion = $(this).val();
        if (!linkedVersion || linkedVersion == url_parts[2]) return;
        var new_url = window.location.href.replace(/\/ui\/[^\/]+/, "/ui/" + linkedVersion);
        window.location.href = new_url;
    });
});
