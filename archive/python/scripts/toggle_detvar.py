#!/usr/bin/env python
"""Comment the detector-variation blocks in and out of the analysis XMLs.

    python python/scripts/toggle_detvar.py off  [PATH ...]   wrap them in XML comments
    python python/scripts/toggle_detvar.py on   [PATH ...]   unwrap them
    python python/scripts/toggle_detvar.py status [PATH ...] report what each XML currently has

With no PATH the script walks every XML under ``xml/`` (recursively, so the nested
``xml/binning_study/<family>/<scheme>/`` files are included) and under ``xml_testing/``.
Give one or more paths to restrict it: a directory is searched recursively, a file is
taken as is.  Paths are tried relative to the current directory first and then relative
to the ma_zexp directory, so both of these work from anywhere::

    python python/scripts/toggle_detvar.py on xml/binning_study
    python python/scripts/toggle_detvar.py status xml/binning_study/asimov/count5

Two regions have to move together: the ``<DetVarFiles>`` block that declares the
per-variation input files, and the ``tag="DetVar"`` ``spline_to_covariance``
systematics that consume them by name.  Commenting out only the first leaves those
systematics referring to variations PROfit no longer knows about.

Each region is wrapped in a plain ``<!--`` / ``-->`` pair on its own lines, which
round-trips exactly: every other comment in these files opens and closes on one
line, so a bare ``<!--`` line is unambiguous.  The wrapped text is checked for
``--``, which XML forbids inside a comment.

``make_binning_study_xmls.py`` copies whatever state its ``xml/nuwro`` sources are in,
so regenerating the binning study XMLs after toggling only the study resets them to
the production state; toggle again afterwards (or toggle everything).
"""

import re
import sys
from pathlib import Path

MA_ZEXP = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = ("xml", "xml_testing")

OPEN = "<!--\n"
CLOSE = "-->\n"

# Matched on whole lines.  The systematic pattern keys on tag="DetVar" alone and is
# indifferent to attribute order, spacing and how many entries there are, so adding or
# aligning variations does not need a change here.
DETVAR_SYSTEMATIC = re.compile(r'^\s*<systematic\b[^>]*\btag="DetVar"[^>]*/>\s*$')
DETVAR_FILES_OPEN = re.compile(r"^\s*<DetVarFiles>\s*$")

REGIONS = (
    # (name, first line of the region, last line of the region)
    ("DetVarFiles", DETVAR_FILES_OPEN, re.compile(r"^\s*</DetVarFiles>\s*$")),
    ("DetVar systematics", DETVAR_SYSTEMATIC, DETVAR_SYSTEMATIC),
)


def resolve_arg(arg):
    """A path argument, taken relative to the cwd if that exists, else to ma_zexp."""
    for candidate in (Path(arg), MA_ZEXP / arg):
        if candidate.exists():
            return candidate.resolve()
    raise SystemExit(f"{arg}: no such file or directory (tried the cwd and {MA_ZEXP})")


def collect_targets(args):
    """Sorted list of XML files under the given paths (default: xml/ and xml_testing/)."""
    roots = [resolve_arg(a) for a in args] if args else [MA_ZEXP / r for r in DEFAULT_ROOTS]
    targets = []
    for root in roots:
        if root.is_file():
            targets.append(root)
        else:
            targets.extend(root.rglob("*.xml"))
    if not targets:
        raise SystemExit("no XML files found")
    return sorted(set(targets))


def display(path):
    """Path relative to ma_zexp when it lives there, otherwise as is."""
    try:
        return path.relative_to(MA_ZEXP)
    except ValueError:
        return path


def find_region(lines, name, start_re, end_re):
    """Line indices [first, last] of the region, or None if it is not present."""
    starts = [i for i, l in enumerate(lines) if start_re.match(l)]
    if not starts:
        return None
    first = starts[0]
    if name == "DetVar systematics":
        # a contiguous run of sibling entries
        last = first
        while last + 1 < len(lines) and end_re.match(lines[last + 1]):
            last += 1
        return first, last
    ends = [i for i, l in enumerate(lines) if end_re.match(l) and i > first]
    if not ends:
        raise ValueError(f"{name}: found the opening tag but no closing tag")
    return first, ends[0]


def already_off(lines):
    """True when the DetVarFiles block is preceded by a bare ``<!--`` line."""
    for i, line in enumerate(lines):
        if DETVAR_FILES_OPEN.match(line):
            return i > 0 and lines[i - 1] == OPEN
    return False


def disable(path):
    lines = path.read_text().splitlines(keepends=True)
    if already_off(lines):
        return "already off"
    # work back to front so the earlier region's indices stay valid
    spans = []
    for name, start_re, end_re in REGIONS:
        span = find_region(lines, name, start_re, end_re)
        if span is None:
            raise ValueError(f"{path}: region {name!r} not found")
        spans.append(span)
    for first, last in sorted(spans, reverse=True):
        body = "".join(lines[first:last + 1])
        if "--" in body:
            raise ValueError(f"{path}: lines {first + 1}-{last + 1} contain '--', "
                             f"which XML forbids inside a comment")
        lines[first:last + 1] = [OPEN, body, CLOSE]
    path.write_text("".join(lines))
    return "off"


def enable(path):
    lines = path.read_text().splitlines(keepends=True)
    if not already_off(lines):
        return "already on"
    spans = []
    for name, start_re, end_re in REGIONS:
        span = find_region(lines, name, start_re, end_re)
        if span is None:
            raise ValueError(f"{path}: region {name!r} not found")
        first, last = span
        if not (first > 0 and lines[first - 1] == OPEN and
                last + 1 < len(lines) and lines[last + 1] == CLOSE):
            raise ValueError(f"{path}: region {name!r} is not wrapped in a bare "
                             f"comment pair; unwrap it by hand")
        spans.append((first, last))
    for first, last in sorted(spans, reverse=True):
        del lines[last + 1]      # the "-->" line
        del lines[first - 1]     # the "<!--" line
    path.write_text("".join(lines))
    return f"on ({len(spans)} regions)"


def status(path):
    lines = path.read_text().splitlines(keepends=True)
    if not any(DETVAR_FILES_OPEN.match(l) for l in lines):
        return "no DetVar block"
    n_var = sum(1 for l in lines if "minimal_detvar_" in l)
    n_sys = sum(1 for l in lines if DETVAR_SYSTEMATIC.match(l))
    return f"{'off' if already_off(lines) else 'on'} ({n_var} files, {n_sys} systematics)"


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    fn = {"off": disable, "on": enable, "status": status}.get(action)
    if fn is None:
        raise SystemExit(__doc__)
    results = {}
    for path in collect_targets(sys.argv[2:]):
        results.setdefault(fn(path), []).append(path)
    for state, paths in sorted(results.items()):
        print(f"{state}: {len(paths)} file(s)")
        for p in paths[:3]:
            print(f"    {display(p)}")
        if len(paths) > 3:
            print(f"    ... and {len(paths) - 3} more")


if __name__ == "__main__":
    main()
