from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from markdown_it import MarkdownIt
from markdown_it.token import Token

_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_VALID_PERCENT_ESCAPE = re.compile(r"%(?:[0-9A-Fa-f]{2})")


class _LiteralDestinationMarkdownIt(MarkdownIt):
    def normalizeLink(self, url: str) -> str:
        return url


_MARKDOWN = _LiteralDestinationMarkdownIt(
    "commonmark",
    options_update={"inline_definitions": True, "store_labels": True},
)
_MARKDOWN.disable("text_join")


@dataclass(frozen=True)
class _MarkdownLink:
    destination: str
    line: int


def _token_destination(token: Token) -> str | None:
    attribute = "src" if token.type == "image" else "href"
    destination = token.attrGet(attribute)
    return destination if isinstance(destination, str) else None


def _has_unescaped_malformed_inline_link(tokens: list[Token]) -> bool:
    state = "search"
    label_depth = 0
    rendered_link_depth = 0

    def consume(content: str, *, delimiters_active: bool) -> bool:
        nonlocal label_depth, state
        for character in content:
            if state == "search":
                if delimiters_active and character == "[":
                    state = "label"
                    label_depth = 1
            elif state == "label":
                if delimiters_active and character == "[":
                    label_depth += 1
                elif delimiters_active and character == "]":
                    label_depth -= 1
                    if label_depth == 0:
                        state = "after_label"
            elif character.isspace() and delimiters_active:
                continue
            elif delimiters_active and character == "(":
                return True
            else:
                state = "search"
                label_depth = 0
                if delimiters_active and character == "[":
                    state = "label"
                    label_depth = 1
        return False

    for token in tokens:
        if token.type == "link_open":
            if rendered_link_depth == 0 and state == "after_label":
                state = "search"
                label_depth = 0
            rendered_link_depth += 1
            continue
        if token.type == "link_close":
            rendered_link_depth = max(0, rendered_link_depth - 1)
            continue
        if rendered_link_depth:
            continue
        if token.type == "image":
            if state == "after_label":
                state = "search"
                label_depth = 0
            continue
        if token.type == "text":
            malformed = consume(token.content, delimiters_active=True)
        elif token.type in {"text_special", "code_inline", "html_inline"}:
            malformed = consume(token.content, delimiters_active=False)
        elif token.type in {"softbreak", "hardbreak"}:
            malformed = consume("\n", delimiters_active=True)
        else:
            continue
        if malformed:
            return True
    return False


def _extract_markdown_links(markdown: str) -> tuple[list[_MarkdownLink], list[str]]:
    environment: dict[str, object] = {}
    tokens = _MARKDOWN.parse(markdown, environment)
    links: list[_MarkdownLink] = []
    errors: list[str] = []
    referenced_labels: set[str] = set()
    definitions: list[Token] = []

    for token in tokens:
        if token.type == "definition":
            definitions.append(token)
            continue
        if token.type != "inline":
            continue

        line = token.map[0] + 1 if token.map else 1
        if re.match(r"^ {0,3}\[[^]\n]+\]:", token.content):
            errors.append(f"line {line}: malformed reference link definition")
        for child in token.children or []:
            if child.type in {"link_open", "image"}:
                destination = _token_destination(child)
                if destination == "":
                    errors.append(f"line {line}: empty link destination")
                elif destination is not None:
                    links.append(_MarkdownLink(destination=destination, line=line))
                label = child.meta.get("label")
                if isinstance(label, str):
                    referenced_labels.add(label)
        if _has_unescaped_malformed_inline_link(token.children or []):
            errors.append(f"line {line}: unterminated link destination")

    skipped_referenced_definitions: set[str] = set()
    for definition in definitions:
        label = definition.meta.get("id")
        destination = definition.meta.get("url")
        line = definition.map[0] + 1 if definition.map else 1
        if destination == "":
            errors.append(f"line {line}: empty link destination")
            continue
        if not isinstance(label, str) or not isinstance(destination, str):
            errors.append(f"line {line}: malformed reference link definition")
            continue
        if label in referenced_labels and label not in skipped_referenced_definitions:
            skipped_referenced_definitions.add(label)
            continue
        links.append(_MarkdownLink(destination=destination, line=line))

    return links, errors


def _strict_url_decode(value: str) -> str:
    invalid_percent = re.sub(_VALID_PERCENT_ESCAPE, "", value)
    if "%" in invalid_percent:
        raise ValueError("invalid percent escape")
    return unquote(value, encoding="utf-8", errors="strict")


def _inline_plain_text(tokens: list[Token]) -> str:
    plain_text: list[str] = []
    for token in tokens:
        if token.type in {"text", "text_special", "code_inline"}:
            plain_text.append(token.content)
        elif token.type in {"softbreak", "hardbreak"}:
            plain_text.append(" ")
        elif token.type == "image":
            plain_text.append(_inline_plain_text(token.children or []))
    return "".join(plain_text)


def _github_slug(heading: str) -> str:
    normalized: list[str] = []
    for character in heading.strip().lower():
        category = unicodedata.category(character)
        if character in {"-", "_"} or character.isspace():
            normalized.append(character)
        elif character.isascii():
            if character.isalnum():
                normalized.append(character)
        elif not category.startswith(("P", "C")):
            normalized.append(character)
    return re.sub(r"\s", "-", "".join(normalized))


def _heading_fragments(markdown: str) -> set[str]:
    tokens = _MARKDOWN.parse(markdown, {})
    headings = [
        _inline_plain_text(inline.children or [])
        for opening, inline in pairwise(tokens)
        if opening.type == "heading_open" and inline.type == "inline"
    ]

    fragments: set[str] = set()
    for heading in headings:
        base = _github_slug(heading)
        fragment = base
        suffix = 0
        while fragment in fragments:
            suffix += 1
            fragment = f"{base}-{suffix}"
        fragments.add(fragment)
    return fragments


def _resolve_exact_path(
    repo: Path, source: Path, decoded_path: str
) -> tuple[Path | None, str | None]:
    if "\x00" in decoded_path:
        return None, "path contains a NUL byte"
    if Path(decoded_path).is_absolute():
        return None, f"path escapes repository: {decoded_path}"

    relative_parts = Path(decoded_path).parts if decoded_path else ()
    current = source.parent if decoded_path else source
    for part in relative_parts:
        if part == ".":
            continue
        if part == "..":
            current = current.parent
        else:
            if not current.is_dir():
                return None, f"parent directory does not exist: {current}"
            exact_names = {entry.name for entry in current.iterdir()}
            if part not in exact_names:
                case_matches = sorted(
                    name for name in exact_names if name.casefold() == part.casefold()
                )
                if case_matches:
                    return None, f"path casing mismatch: {part!r} should be {case_matches[0]!r}"
                return None, f"path does not exist: {current / part}"
            current /= part

        try:
            current.relative_to(repo)
        except ValueError:
            return None, f"path escapes repository: {decoded_path}"

    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(repo.resolve(strict=True))
    except FileNotFoundError:
        return None, f"path does not exist: {current}"
    except ValueError:
        return None, f"path escapes repository: {decoded_path}"
    return current, None


def _validate_documents(repo: Path, documents: list[Path]) -> list[str]:
    repo = repo.resolve(strict=True)
    errors: list[str] = []
    for source in documents:
        try:
            source_label = source.relative_to(repo)
        except ValueError:
            errors.append(f"{source}: source path escapes repository")
            continue
        try:
            resolved_source = source.resolve(strict=True)
            resolved_source.relative_to(repo)
        except FileNotFoundError:
            errors.append(f"{source_label}: source path does not exist")
            continue
        except ValueError:
            errors.append(f"{source_label}: source path escapes repository")
            continue
        if not resolved_source.is_file():
            errors.append(f"{source_label}: source path is not a regular file")
            continue

        markdown = resolved_source.read_text(encoding="utf-8")
        links, syntax_errors = _extract_markdown_links(markdown)
        errors.extend(f"{source_label}:{error}" for error in syntax_errors)

        for link in links:
            destination = link.destination.strip()
            if destination.startswith("//") or _SCHEME.match(destination):
                continue
            try:
                parsed = urlsplit(destination)
                decoded_path = _strict_url_decode(parsed.path)
                decoded_fragment = _strict_url_decode(parsed.fragment)
            except (UnicodeDecodeError, ValueError) as error:
                errors.append(
                    f"{source_label}:line {link.line}: malformed link {destination!r}: {error}"
                )
                continue

            target, path_error = _resolve_exact_path(repo, source, decoded_path)
            if path_error is not None:
                errors.append(f"{source_label}:line {link.line}: {destination!r}: {path_error}")
                continue
            if decoded_fragment:
                assert target is not None
                if target.suffix.casefold() != ".md":
                    errors.append(
                        f"{source_label}:line {link.line}: fragment targets non-Markdown path "
                        f"{destination!r}"
                    )
                    continue
                fragments = _heading_fragments(target.read_text(encoding="utf-8"))
                if decoded_fragment not in fragments:
                    errors.append(
                        f"{source_label}:line {link.line}: heading fragment "
                        f"{decoded_fragment!r} does not exist in {target.relative_to(repo)}"
                    )
    return errors


def _write(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def test_scanner_ignores_fenced_code_inline_code_and_external_schemes(tmp_path: Path) -> None:
    document = _write(
        tmp_path / "README.md",
        """# Real heading
[same document](#real-heading)
[web](https://example.invalid/missing)
[email](mailto:nobody@example.invalid)
`[inline](missing.md)`
```markdown
[fenced](missing.md)
```
""",
    )

    assert _validate_documents(tmp_path, [document]) == []


def test_scanner_ignores_escaped_literal_link_syntax(tmp_path: Path) -> None:
    source = _write(tmp_path / "README.md", "\\[literal](missing.md)\n")

    assert _validate_documents(tmp_path, [source]) == []


@pytest.mark.parametrize(
    "markdown",
    [
        "[literal\\](missing.md)\n",
        "[literal]\\(missing.md)\n",
    ],
)
def test_scanner_keeps_escaped_structural_delimiters_literal(
    tmp_path: Path,
    markdown: str,
) -> None:
    source = _write(tmp_path / "README.md", markdown)

    assert _validate_documents(tmp_path, [source]) == []


@pytest.mark.parametrize("escaped_punctuation", ["*", "_"])
def test_scanner_detects_malformed_destination_after_escape_inside_real_label(
    tmp_path: Path,
    escaped_punctuation: str,
) -> None:
    source = _write(
        tmp_path / "README.md",
        f"[bro\\{escaped_punctuation}ken](missing.md\n",
    )

    errors = _validate_documents(tmp_path, [source])

    assert len(errors) == 1
    assert "unterminated link destination" in errors[0]


def test_scanner_resolves_link_with_escape_inside_real_label(tmp_path: Path) -> None:
    target = _write(tmp_path / "guide.md", "# Guide\n")
    source = _write(tmp_path / "README.md", "[bro\\*ken](guide.md#guide)\n")

    links, syntax_errors = _extract_markdown_links(source.read_text(encoding="utf-8"))

    assert syntax_errors == []
    assert [link.destination for link in links] == ["guide.md#guide"]
    assert _validate_documents(tmp_path, [source, target]) == []


@pytest.mark.parametrize(
    ("markdown", "inner_destination"),
    [
        ("[outer ![alt](image.md)](missing.md\n", "image.md"),
        ("[outer [inner](ok.md)](missing.md\n", "ok.md"),
    ],
)
def test_scanner_detects_malformed_outer_destination_across_valid_nested_atom(
    tmp_path: Path,
    markdown: str,
    inner_destination: str,
) -> None:
    inner = _write(tmp_path / inner_destination, "# Inner\n")
    source = _write(tmp_path / "README.md", markdown)

    links, syntax_errors = _extract_markdown_links(source.read_text(encoding="utf-8"))

    assert [link.destination for link in links] == [inner_destination]
    assert syntax_errors == ["line 1: unterminated link destination"]
    assert _validate_documents(tmp_path, [source, inner]) == [
        "README.md:line 1: unterminated link destination"
    ]


def test_scanner_resolves_real_destinations_with_escaped_punctuation(tmp_path: Path) -> None:
    target = _write(tmp_path / "Guide_(draft).md", "# Draft\n")
    source = _write(tmp_path / "README.md", "[draft](Guide_\\(draft\\).md#draft)\n")

    links, syntax_errors = _extract_markdown_links(source.read_text(encoding="utf-8"))

    assert syntax_errors == []
    assert [link.destination for link in links] == ["Guide_(draft).md#draft"]
    assert _validate_documents(tmp_path, [source, target]) == []


@pytest.mark.parametrize(
    "markdown",
    [
        "```bad`info\n[live](missing.md)\n```\n",
        "```` [live](missing.md) `````\n",
    ],
)
def test_scanner_does_not_mask_live_links_with_invalid_commonmark_delimiters(
    tmp_path: Path,
    markdown: str,
) -> None:
    source = _write(tmp_path / "README.md", markdown)

    errors = _validate_documents(tmp_path, [source])

    assert len(errors) == 1
    assert "path does not exist" in errors[0]


def test_scanner_url_decodes_local_paths_and_heading_fragments(tmp_path: Path) -> None:
    target = _write(tmp_path / "Docs" / "My Guide.md", "# Café notes\n")
    source = _write(tmp_path / "README.md", "[guide](Docs/My%20Guide.md#caf%C3%A9-notes)\n")

    assert target.exists()
    assert _validate_documents(tmp_path, [source, target]) == []


def test_scanner_rejects_paths_that_escape_the_repository(tmp_path: Path) -> None:
    source = _write(tmp_path / "repo" / "README.md", "[outside](../../outside.md)\n")
    _write(tmp_path / "outside.md", "# Outside\n")

    errors = _validate_documents(tmp_path / "repo", [source])

    assert len(errors) == 1
    assert "path escapes repository" in errors[0]


def test_scanner_rejects_wrong_path_casing_even_on_case_insensitive_filesystems(
    tmp_path: Path,
) -> None:
    target = _write(tmp_path / "Docs" / "Guide.md", "# Guide\n")
    source = _write(tmp_path / "README.md", "[guide](docs/guide.md)\n")

    errors = _validate_documents(tmp_path, [source, target])

    assert len(errors) == 1
    assert "path casing mismatch: 'docs' should be 'Docs'" in errors[0]


def test_scanner_accepts_github_duplicate_heading_suffixes(tmp_path: Path) -> None:
    target = _write(tmp_path / "guide.md", "# Repeat\n## Repeat\n## Repeat\n")
    source = _write(tmp_path / "README.md", "[third](guide.md#repeat-2)\n")

    assert _validate_documents(tmp_path, [source, target]) == []


def test_scanner_uses_github_slugs_for_literal_underscores_and_ascii_symbols(
    tmp_path: Path,
) -> None:
    target = _write(tmp_path / "guide.md", "# foo_bar\n# C++\n")
    source = _write(tmp_path / "README.md", "[underscore](guide.md#foo_bar)\n[plus](guide.md#c)\n")

    assert _validate_documents(tmp_path, [source, target]) == []


@pytest.mark.parametrize("wrong_fragment", ["foobar", "c++"])
def test_scanner_rejects_non_github_heading_fragments(
    tmp_path: Path,
    wrong_fragment: str,
) -> None:
    target = _write(tmp_path / "guide.md", "# foo_bar\n# C++\n")
    source = _write(tmp_path / "README.md", f"[wrong](guide.md#{wrong_fragment})\n")

    errors = _validate_documents(tmp_path, [source, target])

    assert len(errors) == 1
    assert f"heading fragment {wrong_fragment!r} does not exist" in errors[0]


def test_scanner_uses_code_span_text_in_github_heading_fragments(tmp_path: Path) -> None:
    target = _write(tmp_path / "guide.md", "# `foo_bar`\n")
    source = _write(tmp_path / "README.md", "[code heading](guide.md#foo_bar)\n")

    assert _validate_documents(tmp_path, [source, target]) == []


def test_scanner_rejects_heading_fragment_that_drops_code_span_underscore(
    tmp_path: Path,
) -> None:
    target = _write(tmp_path / "guide.md", "# `foo_bar`\n")
    source = _write(tmp_path / "README.md", "[wrong](guide.md#foobar)\n")

    errors = _validate_documents(tmp_path, [source, target])

    assert len(errors) == 1
    assert "heading fragment 'foobar' does not exist" in errors[0]


@pytest.mark.parametrize(
    "markdown",
    [
        "[guide][g]\n\n[g]: missing.md\n",
        "![diagram][g]\n\n[g]: missing.md\n",
        "[guide][]\n\n[guide]: missing.md\n",
        "[g]\n\n[g]: missing.md\n",
        "Unused definition.\n\n[g]: missing.md\n",
    ],
)
def test_scanner_validates_local_reference_style_destinations(
    tmp_path: Path,
    markdown: str,
) -> None:
    source = _write(tmp_path / "README.md", markdown)

    errors = _validate_documents(tmp_path, [source])

    assert len(errors) == 1
    assert "path does not exist" in errors[0]


def test_scanner_does_not_reinterpret_a_definition_that_cannot_interrupt_a_paragraph() -> None:
    links, syntax_errors = _extract_markdown_links("[guide][g]\n[g]: missing.md\n")

    assert links == []
    assert syntax_errors == []


def test_scanner_accepts_resolvable_full_collapsed_shortcut_and_image_references(
    tmp_path: Path,
) -> None:
    target = _write(tmp_path / "guide.md", "# Guide\n")
    source = _write(
        tmp_path / "README.md",
        """[full][g]
![image][g]
[collapsed][]
[shortcut]

[g]: guide.md#guide
[collapsed]: guide.md#guide
[shortcut]: guide.md#guide
""",
    )

    links, syntax_errors = _extract_markdown_links(source.read_text(encoding="utf-8"))

    assert syntax_errors == []
    assert [link.destination for link in links] == ["guide.md#guide"] * 4
    assert _validate_documents(tmp_path, [source, target]) == []


def test_scanner_rejects_scanned_source_symlinks_that_escape_the_repository(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = _write(tmp_path / "outside.md", "# Outside\n[self](#outside)\n")
    source = repo / "README.md"
    source.symlink_to(outside)

    errors = _validate_documents(repo, [source])

    assert len(errors) == 1
    assert "source path escapes repository" in errors[0]


@pytest.mark.parametrize(
    ("markdown", "expected_error"),
    [
        ("[missing](missing.md)\n", "path does not exist"),
        ("[empty]()\n", "empty link destination"),
        ("[broken](missing.md\n", "unterminated link destination"),
        ("[bad escape](missing%ZZ.md)\n", "invalid percent escape"),
    ],
)
def test_scanner_reports_unresolvable_and_malformed_local_links(
    tmp_path: Path,
    markdown: str,
    expected_error: str,
) -> None:
    source = _write(tmp_path / "README.md", markdown)

    errors = _validate_documents(tmp_path, [source])

    assert len(errors) == 1
    assert expected_error in errors[0]


def test_repository_markdown_links_are_resolvable_with_exact_casing_and_fragments() -> None:
    repo = Path(__file__).resolve().parents[2]
    readme = repo / "README.md"
    docs_root = repo / "Docs"
    plans_root = repo / "plans"
    assert readme.is_file()
    assert docs_root.is_dir()
    assert plans_root.is_dir()

    documents = [readme, *sorted(docs_root.rglob("*.md")), *sorted(plans_root.rglob("*.md"))]
    errors = _validate_documents(repo, documents)

    assert errors == [], "Broken local Markdown links:\n" + "\n".join(errors)
