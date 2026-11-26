"""Unit tests for src.cleaner."""

from src.cleaner import clean_text, normalize_for_comparison


def test_preserves_technical_terms_and_versions() -> None:
    """Technical symbols (+, #, ., /, -) and versions survive cleaning intact."""
    sample = (
        "Skills: C++, C#, .NET, ASP.NET, Node.js, Vue.js, "
        "CI/CD, TCP/IP, PL/SQL, Python 3.12, full-stack (2022-2025)"
    )
    cleaned = clean_text(sample)
    for token in (
        "C++",
        "C#",
        ".NET",
        "ASP.NET",
        "Node.js",
        "Vue.js",
        "CI/CD",
        "TCP/IP",
        "PL/SQL",
        "3.12",
        "full-stack",
        "2022-2025",
    ):
        assert token in cleaned


def test_removes_ocr_decorative_lines_and_control_chars() -> None:
    """Decorative divider runs, control chars, and broken unicode artifacts are removed."""
    raw = (
        "EXPERIENCE\r\n"
        "________\n"
        "=====\n"
        "-----\n"
        "\x00\x0b\ufeffSenior\u200b  Dev\ufffd\n\n\n\n"
        "Next"
    )
    cleaned = clean_text(raw)
    assert "____" not in cleaned
    assert "====" not in cleaned
    assert "----" not in cleaned
    assert "\x00" not in cleaned
    assert "\ufeff" not in cleaned
    assert "\ufffd" not in cleaned
    assert cleaned == "EXPERIENCE\n\nSenior Dev\n\nNext"


def test_preserves_turkish_characters_and_handles_empty() -> None:
    """Turkish Unicode letters are preserved and whitespace-only text returns empty."""
    assert clean_text("   \n\t  ") == ""
    assert (
        clean_text("Özgeçmiş: Şükrü Çığır, İstanbul")
        == "Özgeçmiş: Şükrü Çığır, İstanbul"
    )


def test_normalize_for_comparison_lowercases_and_collapses_whitespace() -> None:
    """Fair benchmark normalization cleans noise, lowercases, and uses single spaces."""
    raw = "  C++\n\nNode.js\t\t____  CI/CD  "
    assert normalize_for_comparison(raw) == "c++ node.js ci/cd"
