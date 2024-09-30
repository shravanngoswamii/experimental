---
title: "Publications by {{ author }}"
---

# Publications by {{ author }}

{% for entry in entries %}
- {{ entry.title }} ({{ entry.year }})
{% endfor %}
