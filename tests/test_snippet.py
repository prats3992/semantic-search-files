from semantic_search.cli import generate_smart_snippet

CONTENT = (
    "Python is a popular language. "
    "Asyncio lets you schedule coroutines concurrently. "
    "Decorators wrap functions to extend behaviour."
)


def test_picks_sentence_with_most_query_terms():
    snippet = generate_smart_snippet(CONTENT, "schedule coroutines")

    assert snippet == "Asyncio lets you schedule coroutines concurrently."


def test_matching_is_case_insensitive():
    assert "Decorators" in generate_smart_snippet(CONTENT, "DECORATORS")


def test_no_match_falls_back_to_start_of_content():
    content = "x" * 300

    snippet = generate_smart_snippet(content, "unrelated query", snippet_length=50)

    assert snippet == "x" * 50 + "..."


def test_empty_query_returns_start_of_content():
    assert generate_smart_snippet("Short text.", "   ") == "Short text."


def test_long_sentence_is_trimmed_around_match():
    content = "filler " * 100 + "needle " + "filler " * 100

    snippet = generate_smart_snippet(content, "needle", snippet_length=60, context_window=20)

    assert "needle" in snippet
    assert snippet.startswith("...") and snippet.endswith("...")
    assert len(snippet) <= 60 + 6


def test_newlines_are_flattened():
    snippet = generate_smart_snippet("Line one about cats\nand more cats.", "cats")

    assert "\n" not in snippet
