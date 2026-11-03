"""
Smoke tests for functions in main.py that do not require Streamlit runtime or API keys.

Run: pytest tests/test_main_imports.py -v
"""
import sys
import os
import ast
import textwrap
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_function_from_main(func_name: str):
    """
    Extract a named function from main.py without executing Streamlit UI code.
    Uses AST to locate the function definition, then exec() it in a minimal
    namespace — avoids triggering st.set_page_config() at module level.
    """
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"
    )
    source = open(src_path, encoding="utf-8").read()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_src = ast.get_source_segment(source, node)
            if func_src is None:
                raise ImportError(
                    f"Could not extract source for '{func_name}' from main.py"
                )
            namespace = {"re": __import__("re")}
            exec(textwrap.dedent(func_src), namespace)  # noqa: S102
            return namespace[func_name]

    raise ImportError(f"Function '{func_name}' not found in main.py")


# ---------------------------------------------------------------------------
# Test 1: remove_emojis strips emoji characters
# ---------------------------------------------------------------------------
def test_remove_emojis_strips_emojis():
    remove_emojis = _load_function_from_main("remove_emojis")
    result = remove_emojis("Hello 😀 World 🌍")
    assert "😀" not in result
    assert "🌍" not in result
    assert "Hello" in result
    assert "World" in result


# ---------------------------------------------------------------------------
# Test 2: remove_emojis leaves plain text unchanged
# ---------------------------------------------------------------------------
def test_remove_emojis_plain_text():
    remove_emojis = _load_function_from_main("remove_emojis")
    text = "No emojis here at all."
    assert remove_emojis(text) == text


# ---------------------------------------------------------------------------
# Test 3: contains_sensitive_topics returns False for a normal college query
#
# NOTE: sensitive_keywords is intentionally empty in the current codebase
# (the keyword list is a placeholder for future customization). Both tests
# below assert False because there are no keywords to match — this is the
# expected behavior, not a coverage gap to hide. If keywords are added in
# the future, update these tests accordingly.
# ---------------------------------------------------------------------------
def test_contains_sensitive_topics_normal_query():
    contains_sensitive_topics = _load_function_from_main("contains_sensitive_topics")
    assert contains_sensitive_topics("What programs does the college offer?") is False


# ---------------------------------------------------------------------------
# Test 4: contains_sensitive_topics returns False for empty string
# (sensitive_keywords list is empty in the current codebase — see note above)
# ---------------------------------------------------------------------------
def test_contains_sensitive_topics_empty_string():
    contains_sensitive_topics = _load_function_from_main("contains_sensitive_topics")
    assert contains_sensitive_topics("") is False


# ---------------------------------------------------------------------------
# Test 5: contains_sensitive_topics returns True when keywords are present
# Tests the matching logic directly by injecting a non-empty keyword list,
# bypassing the intentionally empty production list.
# ---------------------------------------------------------------------------
def test_contains_sensitive_topics_with_keyword():
    """
    Verify that the matching logic works correctly when keywords are present.
    This test patches sensitive_keywords to a non-empty value so the function
    logic is exercised, rather than vacuously passing on an empty list.
    """
    import importlib
    import ast
    import textwrap
    import types

    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"
    )
    source = open(src_path, encoding="utf-8").read()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "contains_sensitive_topics":
            func_src = ast.get_source_segment(source, node)
            # Replace the empty keyword list with a test keyword
            patched_src = func_src.replace(
                "sensitive_keywords = []",
                'sensitive_keywords = ["forbidden_topic"]',
            )
            namespace = {"re": __import__("re")}
            exec(textwrap.dedent(patched_src), namespace)  # noqa: S102
            fn = namespace["contains_sensitive_topics"]

            # Should match
            assert fn("this is about forbidden_topic details") is True
            # Should not match unrelated query
            assert fn("What courses are available?") is False
            return

    pytest.skip("could not locate contains_sensitive_topics in main.py")
