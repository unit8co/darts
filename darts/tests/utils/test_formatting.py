import textwrap

from darts.utils._formatting import (
    format_bytes,
    format_dict,
    format_list,
    make_collapsible_section,
    make_paragraph,
    truncate_key,
)


class TestFormatting:
    def test_format_bytes(self):
        """Test format_bytes with different units."""
        assert format_bytes(500) == "500.00 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"

    def test_format_bytes_precision(self):
        """Test format_bytes with custom precision."""
        assert format_bytes(1536, precision=1) == "1.5 KB"

    def test_truncate_key(self):
        """Test truncate_key function."""
        assert truncate_key("hello") == "hello"
        assert truncate_key("a" * 20, max_len=20) == "a" * 20
        assert truncate_key("a" * 21, max_len=20) == "a" * 17 + "..."
        assert len(truncate_key("a" * 30, max_len=20)) == 20

    def test_format_dict_empty(self):
        """Test format_dict with empty dict."""
        assert format_dict({}) == "    <empty>"
        assert format_dict({}, render_html=True) == "&lt;empty&gt;"

    def test_format_dict_items(self):
        """Test format_dict with items."""
        d = {"key1": "value1", "key2": "value2"}
        expected = textwrap.indent(
            textwrap.dedent(
                """\
                key1          value1
                key2          value2
                """
            ).rstrip(),
            prefix="    ",
        )
        assert format_dict(d) == expected

    def test_format_dict_truncation(self):
        """Test format_dict truncates long dicts."""
        d = {f"key{i}": f"value{i}" for i in range(10)}
        expected = textwrap.indent(
            textwrap.dedent(
                """\
                key0          value0
                key1          value1
                key2          value2
                key3          value3
                ...           ...
                key9          value9
                """
            ).rstrip(),
            prefix="    ",
        )
        assert format_dict(d) == expected

    def test_format_dict_html(self):
        """Test format_dict HTML rendering."""
        d = {"key": "value"}
        result = format_dict(d, render_html=True)
        assert '<div style="display: flex;' in result
        assert '<div style="min-width: 10em;">key</div>' in result
        assert "<div>value</div>" in result

    def test_format_dict_html_truncation(self):
        """Test format_dict HTML rendering with truncation."""
        d = {f"key{i}": f"value{i}" for i in range(10)}
        result = format_dict(d, max_items=5, render_html=True)
        assert '<div style="min-width: 10em;">...</div>' in result
        assert "<div>...</div>" in result
        assert "key9" in result

    def test_format_list_empty(self):
        """Test format_list with empty list."""
        assert format_list([]) == "<empty>"
        assert format_list([], render_html=True) == "&lt;empty&gt;"

    def test_format_list_items(self):
        """Test format_list with items."""
        assert format_list([1, 2, 3]) == "1, 2, 3"

    def test_format_list_truncation(self):
        """Test format_list truncates long lists."""
        result = format_list(list(range(20)), max_items=5)
        assert result == "0, 1, ..., 17, 18, 19"
        assert result.count("...") == 1

    def test_make_collapsible_section(self):
        """Test make_collapsible_section."""
        result = make_collapsible_section("Title", "Content")
        assert "<details open" in result
        assert "<summary" in result
        assert "Title" in result

    def test_make_collapsible_section_closed(self):
        """Test make_collapsible_section closed."""
        result = make_collapsible_section("Title", "Content", open_by_default=False)
        assert "<details" in result
        assert "open" not in result.split(">")[0]

    def test_make_collapsible_section_flexbox(self):
        """Test make_collapsible_section detects flexbox content."""
        flexbox_content = '<div style="display: flex;">Item</div>'
        result = make_collapsible_section("Title", flexbox_content)
        assert '<div style="margin-left: 0.5em;' in result
        assert "</div>" in result
        assert "<pre" not in result

    def test_make_paragraph(self):
        """Test make_paragraph."""
        result = make_paragraph("Text")
        assert "<p" in result and "</p>" in result
        assert "Text" in result

    def test_make_paragraph_bold(self):
        """Test make_paragraph with bold."""
        result = make_paragraph("Text", bold=True)
        assert "font-weight: bold" in result

    def test_make_paragraph_margin(self):
        """Test make_paragraph with custom margin."""
        result = make_paragraph("Text", margin_left="2em")
        assert "margin-left: 2em" in result
