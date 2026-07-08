def format_bytes(nbytes: int, precision: int = 2) -> str:
    """Formats bytes as human-readable text.

    Parameters
    ----------
    nbytes
        The number of bytes to format.
    precision
        The number of decimal places to round to.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    scale = 1024.0
    i = 0
    while nbytes >= scale and i < len(units) - 1:
        nbytes /= scale
        i += 1
    return f"{nbytes:.{precision}f} {units[i]}"


def truncate_key(key: str, max_len: int = 20) -> str:
    """Truncates a key string to `max_len`, adding '...' if truncated.

    Parameters
    ----------
    key
        The key to truncate.
    max_len
        The maximum length to truncate to.
    """
    key_str = str(key)
    if len(key_str) > max_len:
        return key_str[: max_len - 3] + "..."
    return key_str


def format_dict(
    d: dict,
    max_items: int = 5,
    pad: int = 12,
    render_html: bool = False,
    max_value_len: int = 50,
) -> str:
    """Formats a dictionary as a string, showing at most `max_items` items.
    Keys longer than `pad` are truncated with '...'.
    Values longer than `max_value_len` are truncated with '...'.
    Pass `render_html=True` to use flexbox layout for proper alignment with proportional fonts.

    Parameters
    ----------
    d
        The dictionary to format.
    max_items
        The maximum number of items to show.
    pad
        The number of spaces to pad.
    render_html
        Whether to render HTML output or not.
    max_value_len
        The maximum number of characters to show.
    """
    if not d:
        return "&lt;empty&gt;" if render_html else "    <empty>"

    items = list(d.items())
    show_all = len(items) <= max_items
    items_to_show = items if show_all else items[: max_items - 1]

    s = ""

    # Helper to format a single row
    def format_row(k, v):
        truncated_key = truncate_key(str(k), pad)
        truncated_value = truncate_key(str(v), max_value_len)
        if render_html:
            return (
                f'      <div style="display: flex; margin-left: 1em;">'
                f'<div style="min-width: 10em;">{truncated_key}</div>'
                f"<div>{truncated_value}</div></div>\n"
            )
        else:
            return f"    {truncated_key.ljust(pad)}  {truncated_value}\n"

    for k, v in items_to_show:
        s += format_row(k, v)

    # Add ellipsis row and last item if truncated
    if not show_all:
        if render_html:
            s += (
                '      <div style="display: flex; margin-left: 1em;">'
                '<div style="min-width: 10em;">...</div>'
                "<div>...</div></div>\n"
            )
        else:
            s += f"    {'...'.ljust(pad)}  ...\n"
        last_k, last_v = items[-1]
        s += format_row(last_k, last_v)

    # remove last new line
    s = s.removesuffix("\n")
    return s


def format_list(lst: list, max_items: int = 5, render_html: bool = False) -> str:
    """Formats a list as a string, showing at most `max_items` items.
    Pass `render_html=True` to escape '<' and '>' characters.

    Parameters
    ----------
    lst
        The list to format.
    max_items
        The maximum number of items to show.
    render_html
        Whether to render HTML output or not.
    """
    if not lst:
        if render_html:
            return "&lt;empty&gt;"
        else:
            return "<empty>"

    if len(lst) <= max_items:
        return ", ".join(str(item) for item in lst)

    head = lst[: max_items // 2]
    tail = lst[-(max_items - len(head)) :]
    return (
        ", ".join(str(item) for item in head)
        + ", ..., "
        + ", ".join(str(item) for item in tail)
    )


def make_collapsible_section(
    title: str, content: str, open_by_default: bool = True
) -> str:
    """Creates a collapsible HTML section.

    Parameters
    ----------
    title
        The title of the section.
    content
        The content of the section.
    open_by_default
        Whether to directly open the section when displaying.
    """
    open_tag = " open" if open_by_default else ""
    is_flexbox = '<div style="display: flex;' in content
    wrapper_tag = "div" if is_flexbox else "pre"

    return f"""
    <details{open_tag} style="margin-bottom: 0em;">
        <summary style="font-size: 1em; font-weight: bold; margin-bottom: 0em;">{title}</summary>
        <{wrapper_tag} style="margin-left: 0.5em; font-family: inherit;">{content}</{wrapper_tag}>
    </details>
    """


def make_paragraph(text: str, bold: bool = False, margin_left: str = "0.5em") -> str:
    """Creates an HTML paragraph with optional bold text and margin."""
    style = f"margin-left: {margin_left}; margin-bottom: 1em; text-align: left; font-family: inherit;"
    if bold:
        style += " font-weight: bold;"
    return f"<p style='{style}'>{text}</p>"
