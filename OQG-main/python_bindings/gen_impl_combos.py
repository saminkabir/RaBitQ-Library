#!/usr/bin/env python3
# Generate the AnyImpl variant list + makeImpl + makeImplForLoad block
# (the (numSubspaces, dim) dispatch section) of mn_bindings.hpp from the
# dataset configs in OQG/example/dataset_config.py.
#
# Included combos (deduplicated):
#   1. flexible design : M = suggested_subspaces[ds], dim padded to multiple of M
#   2. fixed-bit ablation: for b in BIT_LIST, M = round(b * dim / 8), dim padded
#
# Usage:
#   python gen_impl_combos.py            # print generated code to stdout
#   python gen_impl_combos.py --apply    # rewrite the block inside mn_bindings.hpp
import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OQG_EXAMPLE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "example")
sys.path.insert(0, OQG_EXAMPLE_DIR)
from dataset_config import datasets, suggested_subspaces

# snapshot before main() mutates suggested_subspaces (e.g. gist -> 768 for
# the large-subspace experiment); the CBCA combos must use the true defaults
ORIG_SUGGESTED = dict(suggested_subspaces)

BINDINGS_PATH = os.path.join(SCRIPT_DIR, "mn_bindings.hpp")

# fixed-bit ablation settings; keep consistent with
# example/fixed_bit_ablation/train_fixed_bit.py
INCLUDE_FIXED_BIT = False
BIT_LIST = [0.5, 1, 2, 3]

INCLUDE_SUGGESTED = True

# extra combos required by other experiments (e.g. large-subspace sensitivity on GIST)
EXTRA_COMBOS = [(128, 1024), (256, 1024), (512, 1024), (1024, 1024)]
EXTRA_COMBOS = [(48, 960)]  # gist default suggested combo (memory_breakdown uses it)

# datasets covered by the CBCA (d = D/4) sensitivity study; the batchSize != 64
# MNGGIndexT instantiations only compile these combos to keep build time low
CBCA_DATASETS = ["gist", "glove", "sift"]

def padded_dim(dim: int, mod: int) -> int:
    return ((dim + mod - 1) // mod) * mod


def subspaces_for_bit(bit: float, dim: int) -> int:
    # 8 bits per PQ subspace; round half up (四舍五入), at least 1
    return max(1, int(bit * dim / 8.0 + 0.5))


def collect_combos(only_datasets=None) -> list[tuple[int, int]]:
    combos = set(EXTRA_COMBOS)
    for name, conf in datasets.items():
        if only_datasets is not None and name not in only_datasets:
            continue
        dim = conf["dim"]

        if INCLUDE_SUGGESTED:
            if name in suggested_subspaces:
                m = suggested_subspaces[name]
                combos.add((m, padded_dim(dim, m)))
            else:
                print(f"// [warn] {name}: not in suggested_subspaces, skipped", file=sys.stderr)

        if INCLUDE_FIXED_BIT:
            for bit in BIT_LIST:
                m = subspaces_for_bit(bit, dim)
                combos.add((m, padded_dim(dim, m)))

    return sorted(combos)  # sorted & deduplicated by (numSubspaces, dim)


def cbca_combos() -> list[tuple[int, int]]:
    combos = set()
    for name in CBCA_DATASETS:
        m = ORIG_SUGGESTED[name]
        combos.add((m, padded_dim(datasets[name]["dim"], m)))
    return sorted(combos)


def gen_code(combos: list[tuple[int, int]]) -> str:
    variant_items = ", ".join(f"Impl<{m},{d}>" for m, d in combos)
    cbca = cbca_combos()
    cbca_items = ", ".join(f"Impl<{m},{d}>" for m, d in cbca)

    # group dims by numSubspaces, keeping sorted order
    def group(cs: list[tuple[int, int]]) -> dict[int, list[int]]:
        by_m: dict[int, list[int]] = {}
        for m, d in cs:
            by_m.setdefault(m, []).append(d)
        return by_m

    def switch_body(by_m: dict[int, list[int]], with_args: bool, pad: str) -> str:
        args = ", M, efC" if with_args else ""
        lines = []
        for m in sorted(by_m):
            lines.append(f"{pad}case {m}:")
            for d in by_m[m]:
                lines.append(
                    f"{pad}    if (dim == {d}) return AnyImpl{{std::in_place_type<Impl<{m},{d}>>{args}}};")
            lines.append(f"{pad}    break;")
        lines.append(f"{pad}default:")
        lines.append(f"{pad}    break;")
        return "\n".join(lines)

    full_m = group(combos)
    cbca_m = group(cbca)

    def make_fn(name: str, sig: str, with_args: bool) -> str:
        return f"""    static AnyImpl {name}({sig}) {{
        if constexpr (useFullComboSet) {{
            switch (numSubspaces) {{
{switch_body(full_m, with_args, ' ' * 12)}
            }}
        }} else {{
            switch (numSubspaces) {{
{switch_body(cbca_m, with_args, ' ' * 12)}
            }}
        }}
        printf("Input (numSubspaces, dim) is (%lu,%lu)\\n", numSubspaces, dim);
        throw std::invalid_argument("Unsupported (numSubspaces, dim) combination");
    }}"""

    return f"""    using FullAnyImpl = std::variant<
        {variant_items}
    >;

    // batchSize != 64 instantiations exist only for the CBCA d=D/4
    // sensitivity study; they compile the sift/glove/gist combos only,
    // keeping template instantiation cost low.
    using CBCAAnyImpl = std::variant<
        {cbca_items}
    >;

    using AnyImpl = std::conditional_t<useFullComboSet, FullAnyImpl, CBCAAnyImpl>;

    AnyImpl impl;

{make_fn("makeImpl", "size_t dim, size_t numSubspaces, size_t M, size_t efC", True)}
{make_fn("makeImplForLoad", "size_t dim, size_t numSubspaces", False)}"""


def apply_to_bindings(code: str):
    with open(BINDINGS_PATH, "r") as f:
        src = f.read()

    start_anchor = "    using FullAnyImpl = std::variant<"
    start = src.find(start_anchor)
    if start < 0:
        # first migration from the old single-list layout
        start_anchor = "    using AnyImpl = std::variant<"
        start = src.find(start_anchor)
    if start < 0:
        raise RuntimeError("start anchor 'using [Full]AnyImpl = std::variant<' not found")

    # end = closing "    }" right after the second (makeImplForLoad's) throw
    throw_anchor = 'throw std::invalid_argument("Unsupported (numSubspaces, dim) combination");'
    first = src.find(throw_anchor, start)
    second = src.find(throw_anchor, first + 1)
    if first < 0 or second < 0:
        raise RuntimeError("throw anchors not found (expect 2 occurrences)")
    end = src.find("\n    }", second)
    if end < 0:
        raise RuntimeError("closing brace of makeImplForLoad not found")
    end += len("\n    }")

    backup = BINDINGS_PATH + ".bak"
    with open(backup, "w") as f:
        f.write(src)

    with open(BINDINGS_PATH, "w") as f:
        f.write(src[:start] + code + src[end:])
    print(f"Applied {BINDINGS_PATH} (backup: {backup})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate AnyImpl combos for mn_bindings.hpp")
    parser.add_argument("--apply", action="store_true",
                        help="rewrite the block in mn_bindings.hpp (default: print to stdout)")
    parser.add_argument("--ds", type=str, default=None,
                        help="comma-separated dataset names; default: all datasets in dataset_config")
    args = parser.parse_args()

    suggested_subspaces['gist'] = 768

    only = None
    if args.ds:
        only = set(args.ds.split(","))
        unknown = only - set(datasets.keys())
        if unknown:
            raise SystemExit(f"unknown datasets: {sorted(unknown)}")

    combos = collect_combos(only)
    print(f"// {len(combos)} unique (numSubspaces, dim) combos "
          f"from {len(only) if only else len(datasets)} datasets "
          f"(suggested={INCLUDE_SUGGESTED}, fixed_bit={BIT_LIST if INCLUDE_FIXED_BIT else None})",
          file=sys.stderr)

    code = gen_code(combos)
    if args.apply:
        apply_to_bindings(code)
    else:
        print(code)


if __name__ == "__main__":
    main()
