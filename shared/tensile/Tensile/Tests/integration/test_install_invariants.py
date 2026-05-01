#!/usr/bin/env python3
################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Tensile multi-arch install regression test.

Runs eight static invariants (A1-A8) against an installed Tensile library
directory. Catches drift between the producer (TensileCreateLibrary), the
on-disk file layout, the consumer-side runtime lookup, and the kpack
routability contract.

See REGRESSION-TEST-SPEC.md for the per-assertion spec, discrimination
mutations, and CI matrix.

Usage:
    python3 test_install_invariants.py --library-dir <path> \\
        --arch <arch> [--arch <arch> ...] \\
        [--baseline-dir <path>]            # enables A4
        [--ordering-dir-a <path> --ordering-dir-b <path>]  # enables A5
        [--allow-unrouted <name> ...] \\
        [--check A1 A2 ... | --skip A1 A2 ...] \\
        [--flavor {tensile,tensilelite,auto}] \\
        [--json <path>]
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import msgpack


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCH_RE = re.compile(r"[_-]gfx[0-9a-z]+(?:[-._]|$)")
PER_ARCH_EXTS = {".hsaco", ".co", ".dat", ".yaml"}

# Files whose contents are confirmed-shared across arches.
DEFAULT_SHARED_ALLOWLIST = {
    "TensileManifest.txt",
    # tensilelite: union of per-arch solution mappings
    "TensileLiteLibrary_lazy_Mapping.dat",
    "TensileLiteLibrary_lazy_Mapping.yaml",
}

# Metadata field names that hold filename references.
REFERENCE_KEYS = {"filenamePrefix", "co_path", "value",
                  "codeObjectFile", "codeObjectFilename"}

# Per-assertion IDs in canonical order.
ALL_ASSERTIONS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dat(path):
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False, strict_map_key=False)


def detect_flavor(library_dir, explicit):
    """Return 'tensile' or 'tensilelite' based on the lazy-master pattern."""
    if explicit != "auto":
        return explicit
    if any(library_dir.glob("TensileLiteLibrary_lazy_*.dat")):
        return "tensilelite"
    return "tensile"


def collect_selector_indices(node, out):
    if isinstance(node, dict):
        if isinstance(node.get("table"), list):
            for row in node["table"]:
                if isinstance(row, dict) and "index" in row:
                    out.add(row["index"])
        for v in node.values():
            collect_selector_indices(v, out)
    elif isinstance(node, list):
        for x in node:
            collect_selector_indices(x, out)


def find_placeholders(node, out):
    """Append `value` strings of every Placeholder node."""
    if isinstance(node, dict):
        if node.get("type") == "Placeholder" and isinstance(node.get("value"), str):
            out.append(node["value"])
        for v in node.values():
            find_placeholders(v, out)
    elif isinstance(node, list):
        for x in node:
            find_placeholders(x, out)


def looks_like_artifact_ref(s):
    if not isinstance(s, str):
        return False
    return ("TensileLibrary" in s or "fallback" in s or "extop_" in s
            or s.endswith((".co", ".hsaco", ".dat", ".yaml")))


def collect_refs(dat_path):
    data = load_dat(dat_path)
    refs = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and k in REFERENCE_KEYS and looks_like_artifact_ref(v):
                    refs.append((str(k), v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(data)
    return refs


def candidates_on_disk(install_dir, ref):
    if ref.endswith((".co", ".hsaco", ".dat", ".yaml")):
        return [install_dir / ref]
    return [install_dir / f"{ref}.co",
            install_dir / f"{ref}.hsaco",
            install_dir / f"{ref}.dat"]


def structural_fingerprint(dat_path):
    """Index/selector fingerprint, excluding solution.name verbosity."""
    data = load_dat(dat_path)
    sols = data.get("solutions", []) or []
    indices = sorted(s["index"] for s in sols if isinstance(s, dict) and "index" in s)
    tables = []
    collect_tables_for_fingerprint(data.get("library", {}), tables)
    return {
        "n_solutions": len(sols),
        "indices": indices,
        "tables": sorted(tables),
    }


def collect_tables_for_fingerprint(node, out):
    """Each selector table → tuple of sorted (key, index) pairs."""
    if isinstance(node, dict):
        if isinstance(node.get("table"), list):
            entries = []
            for row in node["table"]:
                if isinstance(row, dict) and "index" in row:
                    sel_key = row.get("key", row.get("predicate", "?"))
                    entries.append((repr(sel_key), row["index"]))
            if entries:
                out.append(tuple(sorted(entries)))
        for v in node.values():
            collect_tables_for_fingerprint(v, out)
    elif isinstance(node, list):
        for x in node:
            collect_tables_for_fingerprint(x, out)


def logical_name(stem):
    """Drop trailing _gfx<arch> from a .dat filename stem."""
    return re.sub(r"_gfx[0-9a-z]+$", "", stem)


# ---------------------------------------------------------------------------
# Runtime filename construction (A8)
# ---------------------------------------------------------------------------

def construct_filename_tensile(file_prefix, arch, is_source_kernel):
    """Mirror of shared/tensile getCodeObjectFileName, post-fix (idempotent).

    See shared/tensile/Tensile/Source/lib/include/Tensile/PlaceholderLibrary.hpp
    """
    co = file_prefix
    if is_source_kernel:
        if "fallback" in co:
            arch_suffix = "_" + arch
            if not co.endswith(arch_suffix):
                co += arch_suffix
            co += ".hsaco"
        else:
            co += ".hsaco"
    else:
        co += ".co"
    return co


def construct_filename_tensilelite(file_prefix, arch, is_source_kernel):
    """Mirror of tensilelite getCodeObjectFileName.

    Tensilelite has a simpler shape: filePrefix + ".co". No arch logic, no
    source-kernel branch.
    """
    return file_prefix + ".co"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class Result:
    def __init__(self, aid):
        self.id = aid
        self.passed = True
        self.summary = ""
        self.details = []   # list of strings printed under FAIL
        self.metrics = {}   # JSON-friendly counters

    def fail(self, summary, *details):
        self.passed = False
        self.summary = summary
        self.details = list(details)

    def ok(self, summary, **metrics):
        self.summary = summary
        self.metrics.update(metrics)

    def print_human(self, header):
        verdict = "PASS" if self.passed else "FAIL"
        print(f"[{self.id}] {header}")
        print(f"     {verdict}  {self.summary}")
        for d in self.details:
            print(f"           {d}")
        print()

    def to_json(self):
        return {
            "id": self.id,
            "passed": self.passed,
            "summary": self.summary,
            "details": self.details,
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def check_a1_unique_indices(library_dir):
    """Within each .dat, all solution indices are unique."""
    r = Result("A1")
    n_files = 0
    n_dups = 0
    dup_files = []
    for dat in sorted(library_dir.glob("*.dat")):
        n_files += 1
        sols = load_dat(dat).get("solutions", []) or []
        idxs = [s["index"] for s in sols if isinstance(s, dict) and "index" in s]
        dups = len(idxs) - len(set(idxs))
        if dups:
            n_dups += dups
            dup_files.append((dat.name, dups))
    if dup_files:
        r.fail(f"{len(dup_files)} .dat file(s) with duplicate indices ({n_dups} total)",
               *(f"{name}: {n} duplicate(s)" for name, n in dup_files[:10]))
    else:
        r.ok(f"{n_files} .dat files, 0 with duplicate indices", n_files=n_files)
    return r


def check_a2_selector_resolution(library_dir):
    """Every selector index resolves to a declared solution."""
    r = Result("A2")
    n_files = 0
    n_selectors_total = 0
    orphans = []
    for dat in sorted(library_dir.glob("*.dat")):
        n_files += 1
        data = load_dat(dat)
        declared = {s["index"] for s in (data.get("solutions") or [])
                    if isinstance(s, dict) and "index" in s}
        sel_indices = set()
        collect_selector_indices(data.get("library", {}), sel_indices)
        n_selectors_total += len(sel_indices)
        missing = sel_indices - declared
        for idx in missing:
            orphans.append((dat.name, idx))
    if orphans:
        r.fail(f"{len(orphans)} unresolved selector index/file pairs",
               *(f"{name}: selector index {idx} not in solutions[]"
                 for name, idx in orphans[:10]))
    else:
        r.ok(f"{n_selectors_total} selector indices across {n_files} files all resolve",
             n_selectors=n_selectors_total)
    return r


def check_a3_cross_arch_reuse(library_dir):
    """Cross-arch index reuse is benign; no global-index marker exists."""
    r = Result("A3")
    # Forward-looking guard: scan every .dat for a globalIndex-like marker.
    forbidden = ("globalIndex", "global_index", "uniqueIndex", "global_id")
    found_marker = []
    for dat in sorted(library_dir.glob("*.dat")):
        data = load_dat(dat)
        # Recursive scan for forbidden keys.
        def walk(o):
            if isinstance(o, dict):
                for k in o.keys():
                    if k in forbidden:
                        found_marker.append((dat.name, k))
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for x in o:
                    walk(x)
        walk(data)

    # Empirical: per-arch-suffixed siblings with the same logical name should
    # exhibit some index overlap (proves overlap is normal). Skip the empirical
    # check if there's only one .dat per logical group.
    by_logical = {}
    for dat in sorted(library_dir.glob("*.dat")):
        by_logical.setdefault(logical_name(dat.stem), []).append(dat)

    multi_groups = {k: v for k, v in by_logical.items() if len(v) > 1}
    overlap_observed = False
    for group, dats in multi_groups.items():
        sets = []
        for d in dats:
            data = load_dat(d)
            sets.append({s["index"] for s in (data.get("solutions") or [])
                         if isinstance(s, dict) and "index" in s})
        if len(sets) >= 2 and sets[0] & sets[1]:
            overlap_observed = True
            break

    if found_marker:
        r.fail(f"forbidden global-index marker(s) found",
               *(f"{name}: '{k}'" for name, k in found_marker[:10]))
    elif multi_groups and not overlap_observed:
        # No overlap among per-arch siblings is suspicious only in multi-arch
        # installs. Treat as informational, not a hard fail.
        r.ok(f"no forbidden marker; "
             f"{len(multi_groups)} multi-arch group(s) checked, no overlap observed "
             f"(single-arch fixture or unique-per-arch indices)",
             multi_groups=len(multi_groups))
    else:
        r.ok(f"no forbidden marker; cross-arch overlap is tolerated as designed",
             multi_groups=len(multi_groups))
    return r


def check_a4_structural_fingerprint(library_dir, baseline_dir):
    """Rename preserves structural fingerprint per logical group."""
    r = Result("A4")
    if baseline_dir is None:
        r.ok("skipped (no --baseline-dir)")
        return r
    baseline_dats = {}
    for d in baseline_dir.rglob("*.dat"):
        baseline_dats[d.name] = d

    matches = 0
    mismatches = []
    missing = []
    for name, base in sorted(baseline_dats.items()):
        post = library_dir / name
        if not post.exists():
            missing.append(name)
            continue
        fa = structural_fingerprint(base)
        fb = structural_fingerprint(post)
        if (fa["indices"] == fb["indices"]
                and fa["tables"] == fb["tables"]
                and fa["n_solutions"] == fb["n_solutions"]):
            matches += 1
        else:
            mismatches.append((name, fa, fb))
    if mismatches or missing:
        details = []
        for n in missing[:5]:
            details.append(f"{n}: missing in post-rename install")
        for n, fa, fb in mismatches[:5]:
            why = []
            if fa["n_solutions"] != fb["n_solutions"]:
                why.append(f"n_solutions {fa['n_solutions']}→{fb['n_solutions']}")
            if fa["indices"] != fb["indices"]:
                why.append("index set differs")
            if fa["tables"] != fb["tables"]:
                why.append("selector tables differ")
            details.append(f"{n}: {', '.join(why)}")
        r.fail(f"{matches} match(es), {len(mismatches)} structural mismatch(es), "
               f"{len(missing)} missing", *details)
    else:
        r.ok(f"{matches} logical group(s) preserved between baseline and post-rename",
             matches=matches)
    return r


def check_a5_argument_order(dir_a, dir_b):
    """Argument-order independence — every shared .dat byte-identical."""
    r = Result("A5")
    if dir_a is None or dir_b is None:
        r.ok("skipped (need both --ordering-dir-a and --ordering-dir-b)")
        return r
    files_a = {f.name for f in dir_a.glob("*.dat")}
    files_b = {f.name for f in dir_b.glob("*.dat")}
    only_a = files_a - files_b
    only_b = files_b - files_a
    matching = 0
    diverging = []
    for name in sorted(files_a & files_b):
        h_a = hashlib.md5((dir_a / name).read_bytes()).hexdigest()
        h_b = hashlib.md5((dir_b / name).read_bytes()).hexdigest()
        if h_a == h_b:
            matching += 1
        else:
            diverging.append((name, h_a, h_b))
    if only_a or only_b or diverging:
        details = []
        for n in sorted(only_a)[:5]:
            details.append(f"{n}: only in ordering-A")
        for n in sorted(only_b)[:5]:
            details.append(f"{n}: only in ordering-B")
        for n, ha, hb in diverging[:5]:
            details.append(f"{n}: md5 {ha[:8]} != {hb[:8]}")
        r.fail(f"{matching} byte-identical, {len(diverging)} diverging, "
               f"{len(only_a)} only-A, {len(only_b)} only-B", *details)
    else:
        r.ok(f"{matching} .dat file(s) byte-identical across orderings",
             matching=matching)
    return r


def check_a6_kpack_routability(library_dir, allowlist):
    """Every per-arch file has an arch token in its name."""
    r = Result("A6")
    routable = 0
    shared = 0
    unroutable = []
    for f in sorted(library_dir.iterdir()):
        if not f.is_file():
            continue
        if f.name in allowlist:
            shared += 1
            continue
        if f.suffix not in PER_ARCH_EXTS:
            continue
        if ARCH_RE.search(f.name):
            routable += 1
        else:
            unroutable.append(f.name)
    if unroutable:
        r.fail(f"{len(unroutable)} unroutable file(s) "
               f"(routable={routable}, shared={shared}; allowlist={sorted(allowlist)})",
               *unroutable[:10])
    else:
        r.ok(f"routable={routable}, shared={shared}, unroutable=0",
             routable=routable, shared=shared)
    return r


def check_a7_reference_resolution(library_dir):
    """Every embedded .dat reference resolves to a file on disk."""
    r = Result("A7")
    refs_total = 0
    issues = []
    for dat in sorted(library_dir.glob("*.dat")):
        for key, ref in collect_refs(dat):
            refs_total += 1
            cands = candidates_on_disk(library_dir, ref)
            if not any(c.exists() for c in cands):
                issues.append((dat.name, key, ref))
    if issues:
        r.fail(f"{len(issues)} unresolved reference(s) (of {refs_total} total)",
               *(f"{name}: [{key}] -> {ref}" for name, key, ref in issues[:10]))
    else:
        r.ok(f"{refs_total} embedded reference(s) all resolve",
             refs_total=refs_total)
    return r


def normalize_arch(arch):
    """Mirror the C++ runtime: strip ':xnack+'/':xnack-' suffix.

    See shared/tensile/Tensile/Source/lib/include/Tensile/PlaceholderLibrary.hpp
    (getCodeObjectFileName): the runtime takes hardware.archName() and drops
    everything from the first ':' onward before constructing the codeobject
    filename. On-disk filenames therefore use bare tokens ('gfx90a'), with one
    fat codeobject serving both xnack states. This function applies the same
    normalization so callers can pass arches in their canonical form
    ('gfx90a:xnack+') and still get a faithful mirror of runtime behavior.
    """
    pos = arch.find(":")
    return arch[:pos] if pos != -1 else arch


def check_a8_runtime_construction(library_dir, archs, flavor):
    """For each arch, simulate runtime filename construction; assert exists."""
    r = Result("A8")
    construct = (construct_filename_tensilelite if flavor == "tensilelite"
                 else construct_filename_tensile)
    # Both flavors use the same lazy master filename pattern. The flavor
    # only affects how the runtime constructs codeobject filenames from
    # placeholder values (see construct_filename_*). The TensileLite*
    # _Mapping.dat files that disambiguate flavor are sub-indices, not
    # the master.
    lazy_pattern = "TensileLibrary_lazy_{arch}.dat"
    # Normalize+dedupe: the C++ runtime strips xnack before lookup, so
    # 'gfx90a', 'gfx90a:xnack+', 'gfx90a:xnack-' all resolve to the same
    # on-disk artifacts. Without this, an xnack-tagged input would build
    # filenames like '..._gfx90a:xnack+.hsaco' that never exist on disk.
    seen = set()
    archs_normalized = []
    for a in archs:
        n = normalize_arch(a)
        if n not in seen:
            seen.add(n)
            archs_normalized.append(n)
    total_constructed = 0
    missing = []
    archs_checked = []
    for arch in archs_normalized:
        lazy = library_dir / lazy_pattern.format(arch=arch)
        if not lazy.exists():
            r.fail(f"lazy master missing for arch={arch}: {lazy.name}")
            return r
        archs_checked.append(arch)
        prefixes = []
        find_placeholders(load_dat(lazy).get("library", {}), prefixes)
        for prefix in sorted(set(prefixes)):
            sub = library_dir / f"{prefix}.dat"
            if not sub.exists():
                missing.append((arch, prefix, f"{prefix}.dat (placeholder target)"))
                continue
            for sol in (load_dat(sub).get("solutions") or []):
                if not isinstance(sol, dict):
                    continue
                sm = sol.get("sizeMapping", {})
                is_src = bool(sm.get("sourceKernel", False)) if isinstance(sm, dict) else False
                fname = construct(prefix, arch, is_src)
                total_constructed += 1
                if not (library_dir / fname).exists():
                    missing.append((arch, prefix, fname))
    if missing:
        r.fail(f"{len(missing)} constructed filename(s) do not resolve "
               f"(of {total_constructed}; archs={archs_checked})",
               *(f"arch={a}  prefix={p}  built={f}"
                 for a, p, f in missing[:10]))
    else:
        r.ok(f"{total_constructed} constructed filename(s) all resolve "
             f"(archs={archs_checked}, flavor={flavor})",
             constructed=total_constructed, archs=archs_checked, flavor=flavor)
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ASSERTION_HEADERS = {
    "A1": "within each .dat, solution indices are unique",
    "A2": "every selector index resolves to a declared solution",
    "A3": "cross-arch index reuse is benign (per-library scope)",
    "A4": "rename preserves structural fingerprint per logical group",
    "A5": "argument-order independence",
    "A6": "kpack routability is total",
    "A7": "every embedded reference resolves to disk",
    "A8": "runtime constructed-filename simulation",
}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library-dir", type=Path, required=True,
                   help="Installed Tensile library directory")
    p.add_argument("--arch", action="append", default=[],
                   help="GPU arch (e.g. gfx90a); may be repeated")
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="Single-arch baseline install dir (enables A4)")
    p.add_argument("--ordering-dir-a", type=Path, default=None,
                   help="First multi-arch install dir for A5")
    p.add_argument("--ordering-dir-b", type=Path, default=None,
                   help="Second multi-arch install dir for A5")
    p.add_argument("--allow-unrouted", action="append", default=[],
                   help="Add a filename to the kpack shared allowlist")
    p.add_argument("--check", action="append", default=None,
                   help="Run only these assertions (repeatable)")
    p.add_argument("--skip", action="append", default=[],
                   help="Skip these assertions (repeatable)")
    p.add_argument("--flavor", choices=("tensile", "tensilelite", "auto"),
                   default="auto",
                   help="Library flavor (default: auto-detect)")
    p.add_argument("--json", type=Path, default=None,
                   help="Write JSON report to this path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])

    if not args.library_dir.is_dir():
        print(f"ERROR: --library-dir does not exist: {args.library_dir}", file=sys.stderr)
        return 2
    if not args.arch:
        print("ERROR: at least one --arch is required", file=sys.stderr)
        return 2

    selected = list(args.check) if args.check else list(ALL_ASSERTIONS)
    selected = [a for a in selected if a not in args.skip]
    unknown = [a for a in selected if a not in ALL_ASSERTIONS]
    if unknown:
        print(f"ERROR: unknown assertion id(s): {unknown}", file=sys.stderr)
        return 2

    flavor = detect_flavor(args.library_dir, args.flavor)
    allowlist = DEFAULT_SHARED_ALLOWLIST | set(args.allow_unrouted)

    print(f"=== Tensile install invariants ===")
    print(f"library-dir: {args.library_dir}")
    print(f"flavor:      {flavor}")
    print(f"archs:       {args.arch}")
    print(f"checks:      {selected}")
    print()

    runners = {
        "A1": lambda: check_a1_unique_indices(args.library_dir),
        "A2": lambda: check_a2_selector_resolution(args.library_dir),
        "A3": lambda: check_a3_cross_arch_reuse(args.library_dir),
        "A4": lambda: check_a4_structural_fingerprint(args.library_dir, args.baseline_dir),
        "A5": lambda: check_a5_argument_order(args.ordering_dir_a, args.ordering_dir_b),
        "A6": lambda: check_a6_kpack_routability(args.library_dir, allowlist),
        "A7": lambda: check_a7_reference_resolution(args.library_dir),
        "A8": lambda: check_a8_runtime_construction(args.library_dir, args.arch, flavor),
    }

    results = []
    for aid in ALL_ASSERTIONS:
        if aid not in selected:
            continue
        r = runners[aid]()
        results.append(r)
        r.print_human(ASSERTION_HEADERS[aid])

    failed = [r for r in results if not r.passed]
    if args.json:
        report = {
            "library_dir": str(args.library_dir),
            "flavor": flavor,
            "archs": args.arch,
            "results": [r.to_json() for r in results],
            "passed": len(failed) == 0,
        }
        args.json.write_text(json.dumps(report, indent=2))

    print(f"=== Summary ===")
    print(f"  passed: {len(results) - len(failed)} / {len(results)}")
    if failed:
        print(f"  failed: {[r.id for r in failed]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
