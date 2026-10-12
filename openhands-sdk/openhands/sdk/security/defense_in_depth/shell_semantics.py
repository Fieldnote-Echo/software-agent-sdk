"""Shared AST-backed shell command semantics for security analyzers.

Resolves command names and arguments against the shared tree-sitter
``_shell_ast`` view (issue #2721). Policy rails reuse these facts for pipeline,
wrapper, raw-disk, delete, and nested-script analysis. The pattern analyzer
uses the same view for recursive force-delete detection. Injection and
Python-code patterns stay as regex in ``pattern.py`` because they are not shell
syntax.

The destructive-command scanner reports ``uncertain`` when it cannot verify a
relevant structure. The analyzer can then emit ``UNKNOWN`` rather than a false
``LOW``:

- the destructive flag shape appears on a command whose verb cannot be
  resolved;
- a dynamic argument or unsupported option group prevents safe selection of a
  shell runner's script operand;
- an ERROR/MISSING parse node intersects a command span the detector uses; or
- the nested-runner depth bound is hit with a script operand still unscanned.

Shell parse errors outside these spans stay ``LOW``. Arbitrary non-shell text
can contain parse errors, so mapping each error to ``UNKNOWN`` would produce too
many uncertain results. Detector IDs are owned by ``pattern.py`` and passed in
unchanged.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Final

from tree_sitter import Node

from openhands.sdk.security import _shell_ast as ast_view
from openhands.sdk.security._shell_ast import (
    ShellCommand,
    ShellProgram,
    ShellWord,
)


# Command basenames whose script operand is a nested shell program.
# Resolving and re-parsing that operand closes the nested-runner bypass
# class without hardcoding payloads.
_SHELL_RUNNERS: Final[frozenset[str]] = frozenset(
    {"sh", "bash", "dash", "zsh", "ksh", "ash"}
)

# Basename of the recursive force-delete command resolved structurally.
_RM_BASENAME: Final[str] = "rm"

# The short-option character that makes a runner's first operand a script.
_SCRIPT_FLAG_CHAR: Final[str] = "c"

# End-of-options marker that terminates option parsing (POSIX).
_END_OF_OPTIONS: Final[str] = "--"

# A lone dash: the historical end-of-options synonym POSIX sh still accepts.
_BARE_DASH: Final[str] = "-"

# Leading characters that mark a runner argv word as an option group. POSIX
# sh accepts both ``-x``-style and ``+x``-style groups.
_OPTION_PREFIXES: Final[frozenset[str]] = frozenset({"-", "+"})

# Runner short options that consume the next word as their argument (set -o /
# shopt -O names). Without this, ``bash -o errexit -c ...`` would misread the
# option argument as the first operand and stop looking for the script flag.
_SHORT_OPTS_WITH_ARG: Final[frozenset[str]] = frozenset({"o", "O"})

# Runner long options that consume the next word as their argument. Exotic
# long options beyond these are out of scope: an unrecognized long option is
# treated as argumentless, which at worst misreads its argument as the first
# operand and stops descending -- it never over-descends.
_LONG_OPTS_WITH_ARG: Final[frozenset[str]] = frozenset({"--rcfile", "--init-file"})

# Long-flag names mapped onto their short-flag characters below.
_LONG_RECURSIVE: Final[str] = "recursive"
_LONG_FORCE: Final[str] = "force"

# Node types that make a command name dynamic (value known only at runtime).
# Any such child makes the name unresolvable at analysis time, so resolution
# fails and the caller falls back.
_DYNAMIC_NAME_NODE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "command_substitution",
        "process_substitution",
        "expansion",
        "simple_expansion",
        "arithmetic_expansion",
        "ansi_c_string",
    }
)

# How deep nested script operands are followed. Bounds work on adversarial
# deeply-nested runner arguments; hitting the bound with an operand still
# unscanned is surfaced as uncertain, never a silent LOW.
_MAX_NESTING_DEPTH: Final[int] = 8

# Characters a backslash escapes inside double quotes (POSIX 2.2.3). Before
# any other character the backslash is literal and must be retained.
_DOUBLE_QUOTE_ESCAPABLE: Final[frozenset[str]] = frozenset('$`"\\')

# Quoted dynamic content stays in one argv position, except for the Bash
# ``@`` expansions handled below. Other unresolved word shapes can split.
_SINGLE_ARGUMENT_DYNAMIC_NODE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "string",
        "raw_string",
        "ansi_c_string",
        "arithmetic_expansion",
        "process_substitution",
    }
)

_DEFINITE_NON_OPTION_NODE_TYPES: Final[frozenset[str]] = frozenset(
    {"arithmetic_expansion", "process_substitution"}
)


@dataclass(frozen=True, slots=True)
class CommandSemantics:
    """Static argv facts for the effective command."""

    basename: str | None
    arguments: tuple[str | None, ...]
    argument_words: tuple[ShellWord, ...]
    uncertain: bool = False


@dataclass(frozen=True, slots=True)
class ShellScanResult:
    """Outcome of an AST-backed shell scan.

    ``detector_id`` is the matched pattern stable ID when ``matched`` is True,
    else None. ``uncertain`` is True when the scan cannot resolve a relevant
    command or shell-runner boundary, cannot trust a required parse span, or
    reaches the nesting depth bound. The caller then emits ``UNKNOWN`` instead
    of ``LOW``.
    """

    matched: bool
    detector_id: str | None = None
    uncertain: bool = False


@dataclass(frozen=True, slots=True)
class SignalOutput:
    """Outcome of scanning one program or command for a destructive verb.

    ``matched`` is True once a recursive force delete is resolved concretely.
    ``uncertain`` is True when the scan cannot verify a destructive shape or a
    relevant shell-runner boundary. The caller turns this into ``UNKNOWN``
    rather than ``LOW``.

    A concrete match always outranks uncertainty, so ``matched`` and
    ``uncertain`` are never both True.
    """

    matched: bool
    uncertain: bool


_NO_SIGNAL: Final[SignalOutput] = SignalOutput(matched=False, uncertain=False)
_MATCH_SIGNAL: Final[SignalOutput] = SignalOutput(matched=True, uncertain=False)
_UNCERTAIN_SIGNAL: Final[SignalOutput] = SignalOutput(matched=False, uncertain=True)


def scan_shell_command(command: str, rm_detector_id: str) -> ShellScanResult:
    """Scan ``command`` for AST-resolvable destructive shell commands.

    Resolves each simple command name through quotes and path prefixes,
    descends into shell-runner script operands, and reports a match for the
    recursive force-delete family.

    Uncertainty is reported narrowly. It applies when the scan cannot verify a
    destructive command, a relevant shell-runner option boundary, a required
    parse span, or a nested script before the depth limit. In each case the
    caller emits UNKNOWN rather than a false LOW. An unrelated parse error
    stays LOW.
    """
    program = _safe_parse(command)
    if program is None:
        # Encoding failure (e.g. lone surrogate). The regex layer already
        # scanned the flattened text; report neither match nor uncertainty.
        return ShellScanResult(matched=False)

    signal = _program_signal(program, depth=0)
    if signal.matched:
        return ShellScanResult(matched=True, detector_id=rm_detector_id)
    return ShellScanResult(matched=False, uncertain=signal.uncertain)


def _safe_parse(command: str) -> ShellProgram | None:
    with contextlib.suppress(UnicodeEncodeError, ValueError):
        return ast_view.parse_shell_program(command)
    return None


def resolve_command(command: ShellCommand) -> CommandSemantics:
    """Resolve a command name and its literal arguments."""
    name_literal = (
        _resolve_node_literal(command.name.node) if command.name is not None else None
    )
    name_arity_uncertain = command.name is not None and argument_arity_is_uncertain(
        name_literal,
        command.name,
    )
    return CommandSemantics(
        basename=(
            _posix_basename(name_literal)
            if name_literal is not None and not name_arity_uncertain
            else None
        ),
        arguments=tuple(_resolve_node_literal(word.node) for word in command.words),
        argument_words=command.words,
        uncertain=(
            command.name is not None and (name_literal is None or name_arity_uncertain)
        ),
    )


def has_recursive_force_flags(command: CommandSemantics) -> bool:
    """Return whether a resolved command has recursive and force flags."""
    flags = _collect_literal_flags(command.arguments)
    return ("r" in flags or "R" in flags) and "f" in flags


def static_shell_script(command: CommandSemantics) -> tuple[bool, str | None, bool]:
    """Return the script and option-boundary uncertainty for a shell runner."""
    if command.basename not in _SHELL_RUNNERS:
        return False, None, False
    return _extract_script_operand(command.arguments, command.argument_words)


def argument_arity_is_uncertain(
    literal: str | None,
    word: ShellWord,
) -> bool:
    """Return whether one source word can change the number of argv words."""
    if literal is not None:
        return _contains_unquoted_arity_syntax(word.text)
    return _node_arity_is_uncertain(word.node)


def static_word_prefix(word: ShellWord) -> str:
    """Return the literal prefix before the first dynamic node in a shell word."""
    prefix, _ = _static_node_prefix(word.node)
    return prefix


def _static_node_prefix(node: Node) -> tuple[str, bool]:
    if node.type in _DYNAMIC_NAME_NODE_TYPES:
        return "", False
    if node.type in {"string", "command_name", "concatenation"}:
        parts: list[str] = []
        for child in node.named_children:
            prefix, complete = _static_node_prefix(child)
            parts.append(prefix)
            if not complete:
                return "".join(parts), False
        return "".join(parts), True
    literal = _resolve_node_literal(node)
    return (literal, True) if literal is not None else ("", False)


def _node_arity_is_uncertain(node: Node) -> bool:
    if node.type == "string":
        return _contains_multi_argument_expansion(node)
    if node.type in _SINGLE_ARGUMENT_DYNAMIC_NODE_TYPES or node.type in {
        "word",
        "number",
    }:
        return False
    if node.type == "concatenation":
        return _contains_unquoted_arity_syntax(_decode(node.text)) or any(
            _node_arity_is_uncertain(child) for child in node.named_children
        )
    return True


def _contains_multi_argument_expansion(node: Node) -> bool:
    if node.type in {"expansion", "simple_expansion"}:
        if any(
            child.type == "special_variable_name" and _decode(child.text) == "@"
            for child in node.named_children
        ):
            return True
        if node.type == "expansion":
            subscript = next(
                (child for child in node.named_children if child.type == "subscript"),
                None,
            )
            if subscript is not None:
                index = subscript.child_by_field_name("index")
                if index is not None and _decode(index.text) == "@":
                    return True
            child_types = [child.type for child in node.children]
            if "!" in child_types and child_types[-2:] == ["@", "}"]:
                return True
    return any(
        _contains_multi_argument_expansion(child) for child in node.named_children
    )


def _contains_unquoted_arity_syntax(text: str) -> bool:
    """Detect unquoted pathname and brace expansion syntax in one shell word."""
    bracket_open = False
    brace_expands: list[bool] = []
    quote: str | None = None
    index = 0
    while index < len(text):
        character = text[index]
        if quote == "'":
            if character == quote:
                quote = None
            index += 1
            continue
        if quote == '"':
            if (
                character == "\\"
                and index + 1 < len(text)
                and (
                    text[index + 1] in _DOUBLE_QUOTE_ESCAPABLE
                    or text[index + 1] == "\n"
                )
            ):
                index += 2
                continue
            if character == quote:
                quote = None
            index += 1
            continue
        if character == "\\":
            index += 2
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in {"*", "?"}:
            return True
        elif character == "[":
            bracket_open = True
        elif character == "]" and bracket_open:
            return True
        elif character == "{":
            brace_expands.append(False)
        elif character == "}" and brace_expands:
            if brace_expands.pop():
                return True
        elif brace_expands and (character == "," or text.startswith("..", index)):
            brace_expands[-1] = True
        index += 1
    return False


def _program_signal(program: ShellProgram, depth: int) -> SignalOutput:
    """Return the scan signal for a parsed program.

    ``matched`` is True when a recursive force delete is resolved. ``uncertain``
    is True when the destructive flag shape is present on a command whose verb
    cannot be resolved statically.
    """
    uncertain = False
    for command in ast_view.iter_commands(program):
        resolved = resolve_command(command)
        if has_recursive_force_flags(resolved):
            if resolved.basename == _RM_BASENAME:
                return _MATCH_SIGNAL
            if resolved.basename is None or command.has_error:
                # Dangerous flag shape on a verb we cannot vouch for:
                # either the name is unresolvable, or an ERROR/MISSING
                # parse node intersects this command, so the resolved verb
                # may not be the verb the shell would run. Keep scanning
                # for a concrete match first.
                uncertain = True
        elif resolved.basename == _RM_BASENAME and any(
            literal is not None and argument_arity_is_uncertain(literal, word)
            for literal, word in zip(
                resolved.arguments,
                resolved.argument_words,
                strict=True,
            )
        ):
            uncertain = True
        nested = _nested_signal(command, resolved, depth)
        if nested.matched:
            return _MATCH_SIGNAL
        uncertain = uncertain or nested.uncertain
    return SignalOutput(matched=False, uncertain=uncertain)


def _collect_literal_flags(arguments: tuple[str | None, ...]) -> set[str]:
    """Collect short-flag chars and recognised long flags.

    Option words are resolved through the same quote-removal path as the
    command name. POSIX quote removal makes ``rm "-rf" /`` and ``rm -r"f" /``
    argv-identical to ``rm -rf /``, so treating a quoted option word as opaque
    -- and therefore flagless -- would let the same catastrophic delete through
    under a different spelling while the de-quoted verb still resolved to
    ``rm``. Resolving the verb but not its options is the asymmetry that made
    that bypass possible.

    A word that is dynamic (command substitution, expansion, ANSI-C string)
    stays unresolved and contributes no flags. That matches how an
    unresolvable script operand is treated in ``static_shell_script()``:
    passing flags through a variable is ordinary benign usage, and flagging it
    would flood the ensemble with UNKNOWNs.

    The end-of-options marker terminates flag parsing, and it is compared
    against the de-quoted literal too -- after ``rm "--"`` the shell really
    does treat a following ``-rf`` as a filename. Long flags are mapped onto
    their short-flag character so ordering and mixing work.
    """
    flags: set[str] = set()
    for literal in arguments:
        if literal is None:
            continue
        if literal == _END_OF_OPTIONS:
            break
        flags |= ast_view.split_short_flags(literal)
        if ast_view.is_long_flag(literal, _LONG_RECURSIVE):
            flags.add("r")
        elif ast_view.is_long_flag(literal, _LONG_FORCE):
            flags.add("f")
    return flags


def _nested_signal(
    command: ShellCommand,
    resolved: CommandSemantics,
    depth: int,
) -> SignalOutput:
    """Descend into a shell runner script operand and re-parse it.

    Closes the nested-runner bypass class, including quoted verbs inside the
    script that no outer pattern can see.

    Fail-safe ordering: the operand scan runs first so a concrete match
    always wins; only when no match is found does a parse error on the
    runner's own node force uncertainty -- an ERROR/MISSING there (e.g. an
    unclosed quote truncating the script operand) means the argv spans the
    scan just read cannot be trusted, so a silent LOW would vouch for
    something never actually seen.
    """
    if resolved.basename not in _SHELL_RUNNERS:
        return _NO_SIGNAL
    has_script, inner, uncertain = static_shell_script(resolved)
    if has_script and inner is not None:
        if depth >= _MAX_NESTING_DEPTH:
            return _UNCERTAIN_SIGNAL
        inner_program = _safe_parse(inner)
        if inner_program is not None:
            nested = _program_signal(inner_program, depth + 1)
            if nested.matched:
                return _MATCH_SIGNAL
            if nested.uncertain:
                return _UNCERTAIN_SIGNAL
    if uncertain:
        return _UNCERTAIN_SIGNAL
    if command.has_error:
        return _UNCERTAIN_SIGNAL
    return _NO_SIGNAL


def _extract_script_operand(
    arguments: tuple[str | None, ...],
    argument_words: tuple[ShellWord, ...],
) -> tuple[bool, str | None, bool]:
    """Locate the runner script operand using POSIX argv option semantics.

    Short options combine, so the script flag counts at any position in a
    group (``-xc``, ``-cx``); an argument-taking option consumes the next
    word; ``--`` (or the historical lone ``-``) terminates option parsing;
    and the first operand ends the option list. With the script flag set,
    that first operand IS the script. Without it, the first operand is a
    script *file*, and any flag word after it -- or after ``--`` -- belongs
    to that script rather than to the runner, so it must not trigger a
    descent.

    Returns ``(has_flag, operand, uncertain)``. ``uncertain`` is True when a
    dynamic word or unsupported option group prevents safe selection of the
    script operand.
    """
    has_script_flag = False
    options_ended = False
    index = 0
    while index < len(arguments):
        literal = arguments[index]
        word = argument_words[index]
        if argument_arity_is_uncertain(literal, word):
            if options_ended and not has_script_flag:
                return False, None, False
            return has_script_flag, None, True
        if literal is None:
            if has_script_flag:
                return True, None, False
            if options_ended:
                return False, None, False
            if word.node_type in _DEFINITE_NON_OPTION_NODE_TYPES:
                return False, None, False
            return False, None, index + 1 < len(arguments)
        if not options_ended:
            if literal == _END_OF_OPTIONS or literal == _BARE_DASH:
                options_ended = True
                index += 1
                continue
            if _is_option_word(literal):
                if _sets_script_flag(literal):
                    has_script_flag = True
                if not literal.startswith(_END_OF_OPTIONS) and any(
                    flag in _SHORT_OPTS_WITH_ARG for flag in literal[1:-1]
                ):
                    return has_script_flag, None, True
                if _consumes_next_word(literal):
                    value_index = index + 1
                    if value_index < len(arguments) and (
                        argument_words[value_index].node_type
                        in _DEFINITE_NON_OPTION_NODE_TYPES
                        or argument_arity_is_uncertain(
                            arguments[value_index],
                            argument_words[value_index],
                        )
                    ):
                        return has_script_flag, None, True
                    index += 2
                else:
                    index += 1
                continue
        # First operand reached.
        if has_script_flag:
            return True, literal, False
        return False, None, False
    return has_script_flag, None, False


def _is_option_word(literal: str) -> bool:
    """Return whether a resolved argv word is an option group."""
    return len(literal) > 1 and literal[0] in _OPTION_PREFIXES


def _sets_script_flag(literal: str) -> bool:
    """Return whether a short-option group contains the script flag.

    A single ``-`` or ``+`` prefix starts a short-option group. Repeated
    prefixes do not.
    """
    return (
        len(literal) > 1
        and literal[0] in _OPTION_PREFIXES
        and literal[1] not in _OPTION_PREFIXES
        and _SCRIPT_FLAG_CHAR in literal[1:]
    )


def _consumes_next_word(literal: str) -> bool:
    """Return whether an option word takes the next word as its argument.

    A final short ``o`` or ``O`` consumes the next word. The caller treats an
    earlier argument-taking flag as an unsupported option boundary.
    """
    if literal in _LONG_OPTS_WITH_ARG:
        return True
    if literal.startswith(_END_OF_OPTIONS):
        return False
    return literal[-1] in _SHORT_OPTS_WITH_ARG


def _resolve_node_literal(node: Node) -> str | None:
    """Concatenate literal text from a name/word node, or None if dynamic.

    Walks the concatenation/string/raw_string/word structure that
    tree-sitter-bash produces for command names and arguments. Any dynamic
    child (substitution, expansion, ANSI-C string) makes the value
    unresolvable at analysis time and yields None.
    """
    parts = _collect_literal_parts(node)
    if parts is None:
        return None
    return "".join(parts)


def _collect_literal_parts(node: Node) -> list[str] | None:
    """Return the literal fragments of ``node``, or None if it is dynamic.

    None means the subtree is not statically resolvable. Fragments are
    escape-normalized to the effective literal the shell would produce after
    quote removal: tree-sitter keeps backslash escapes inline in ``word`` and
    ``string_content`` text, so comparing raw text would let ``r\\m`` (which
    the shell runs as ``rm``) slip past the basename check.
    """
    if node.type in _DYNAMIC_NAME_NODE_TYPES:
        return None

    match node.type:
        case "word" | "number":
            return [_unescape_unquoted(_decode(node.text))]

        case "raw_string":
            # Single-quoted operand: strip the quotes; contents are literal
            # and backslash has no special meaning inside single quotes.
            text = _decode(node.text)
            return [text[1:-1] if len(text) >= 2 else ""]

        case "string_content":
            return [_unescape_double_quoted(_decode(node.text))]

        # "string" is the double-quoted case: recursing catches an embedded
        # expansion as dynamic while string_content fragments contribute
        # literally. command_name and concatenation recurse for the same
        # reason, so all three share one branch.
        case "string" | "command_name" | "concatenation":
            parts: list[str] = []
            for child in node.named_children:
                child_parts = _collect_literal_parts(child)
                if child_parts is None:
                    return None
                parts.extend(child_parts)
            return parts

        # Unknown / unhandled node type in a name position: treat as dynamic
        # so we never fabricate a resolved name we are not sure about.
        case _:
            return None


def _unescape_unquoted(text: str) -> str:
    """Strip backslash escapes from an unquoted word (POSIX 2.2.1).

    Outside quotes a backslash makes the next character literal, so the
    effective value drops the backslash (``r\\m`` -> ``rm``); a
    backslash-newline pair is a line continuation and disappears entirely.
    """
    return _strip_backslash_escapes(text, escapable=None)


def _unescape_double_quoted(text: str) -> str:
    """Strip the escapes double quotes honour (POSIX 2.2.3).

    Inside double quotes a backslash only escapes ``$``, `````, ``"``,
    ``\\`` and newline; before any other character the backslash itself is
    literal and must be retained (``"r\\m"`` stays ``r\\m``).
    """
    return _strip_backslash_escapes(text, escapable=_DOUBLE_QUOTE_ESCAPABLE)


def _strip_backslash_escapes(text: str, escapable: frozenset[str] | None) -> str:
    """Return ``text`` with backslash escapes reduced to their literals.

    ``escapable`` is the set of characters whose escapes are honoured, or
    None to honour every character. Backslash-newline is always a line
    continuation and yields nothing. A trailing lone backslash is kept.
    """
    parts: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\" and index + 1 < length:
            follower = text[index + 1]
            if follower == "\n":
                index += 2
                continue
            if escapable is None or follower in escapable:
                parts.append(follower)
                index += 2
                continue
        parts.append(char)
        index += 1
    return "".join(parts)


def _posix_basename(path: str) -> str:
    """Return the final path segment (basename) without importing posixpath.

    A trailing slash yields an empty basename, matching POSIX ``basename``
    semantics closely enough for command-name resolution.
    """
    return path.rsplit("/", 1)[-1]


def _decode(raw: bytes | None) -> str:
    return raw.decode() if raw is not None else ""
