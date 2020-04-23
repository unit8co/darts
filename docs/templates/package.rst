{%- macro automodule(modname, options) -%}
.. automodule:: {{ modname }}
{%- for option in options %}
   :{{ option }}:
{%- endfor %}
{%- endmacro %}

{%- macro toctree(docnames, maxdepth) -%}
.. toctree::
   :maxdepth: {{ maxdepth }}
{% for docname in docnames %}
   {{ docname }}
{%- endfor %}
{%- endmacro %}

{%- if is_namespace %}
{{- [pkgname, "namespace"] | join(" ") | e | heading }}
{% else %}
{{- [pkgname, "package"] | join(" ") | e | heading }}
{% endif %}

{%- if modulefirst and not is_namespace %}
{{ automodule(pkgname, automodule_options) }}
{% endif %}

{%- if subpackages %}
Subpackages
-----------

{{ toctree(subpackages, 1) }}
{% endif %}

{%- if submodules %}
Submodules
----------
{% if separatemodules %}
{{ toctree(submodules, 1) }}
{%- else %}
{%- for submodule in submodules %}
{% if show_headings %}
{{- [submodule, "module"] | join(" ") | e | heading(2) }}
{% endif %}
{{ automodule(submodule, automodule_options) }}
{% endfor %}
{%- endif %}
{% endif %}