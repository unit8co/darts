def format_bytes(nbytes: int, precision: int = 2) -> str:
    """Formats bytes as human-readable text."""
    units = ["B", "KB", "MB", "GB", "TB"]
    scale = 1024.0
    i = 0
    while nbytes >= scale and i < len(units) - 1:
        nbytes /= scale
        i += 1
    return f"{nbytes:.{precision}f} {units[i]}"


def truncate_key(key: str, max_len: int = 20) -> str:
    """Truncates a key string to `max_len`, adding '...' if truncated."""
    key_str = str(key)
    if len(key_str) > max_len:
        return key_str[: max_len - 3] + "..."
    return key_str


def format_dict(
    d: dict, max_items: int = 5, pad: int = 12, render_html: bool = False
) -> str:
    """Formats a dictionary as a string, showing at most `max_items` items.
    Keys longer than `key_max_len` are truncated with '...'.
    Pass `render_html=True` to escape '<' and '>' characters.
    """
    if not d:
        if render_html:
            return "&lt;empty&gt;"
        else:
            return "    <empty>"

    s = ""
    items = list(d.items())

    if len(items) <= max_items:
        for k, v in items:
            s += f"    {truncate_key(k, pad).ljust(pad)}   {v}\n"
    else:
        head = items[: max_items // 2]
        tail = items[-(max_items - len(head)) :]  # keep total items <= max_items

        for k, v in head:
            s += f"    {truncate_key(k, pad).ljust(pad)}   {v}\n"
        s += "    ...\n"
        for k, v in tail:
            s += f"    {truncate_key(k, pad).ljust(pad)}   {v}\n"

    return s


def format_list(lst: list, max_items: int = 5, render_html: bool = False) -> str:
    """Formats a list as a string, showing at most `max_items` items.
    Pass `render_html=True` to escape '<' and '>' characters.
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
    """Creates a collapsible HTML section."""
    open_tag = " open" if open_by_default else ""
    return f"""
    <details{open_tag}>
        <summary style="font-size: 1.2em;">{title}</summary>
        <pre style="margin-left: 0.5em; font-family: inherit;">{content}</pre>
    </details>
    """


def make_paragraph(
    text: str, bold: bool = False, size: str = "1.2em", margin_left: str = "0.5em"
) -> str:
    """Creates an HTML paragraph with optional bold text, custom font size, and margin."""
    if bold:
        text = f"<strong>{text}</strong>"
    # Use margin_left parameter in the style
    style = f"margin-left: {margin_left}; font-size: {size}; text-align: left;"
    return f"<p style='{style}'>{text}</p>"
