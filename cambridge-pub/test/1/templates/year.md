---
title: "Publications in {{ year }}"
---

# Publications in {{ year }}

{% for entry in entries %}
- {{ entry.title }} by {{ entry.author }}
{% endfor %}
