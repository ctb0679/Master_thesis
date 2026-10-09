#!/usr/bin/env python3
"""
clean_ifc.py — command-line entry point of the IFC-to-JSON extractor.

    python3 clean_ifc.py --input path/to/building.ifc --output-dir data/cleaned_ifc
    # -> building_full.json (everything) + building_llm.json (compact view)

The work is done by the ifc_extract package next to this file (see its
__init__.py for the module map). Contract: only ifcopenshell, shapely, numpy
and the standard library; nothing outside this folder is imported.

What is exported: every rooted IFC entity except a short blacklist
(annotation, grids, structural analysis), keyed by GlobalId, with attributes,
psets/qtos (type-inherited), type, materials, appearance, classifications,
folded relations and geometry, all in m / m² / m³ / degrees. Nothing is
dropped silently: skips, drops, geometric guesses and caught exceptions are
counted in the "audit" key, which is also printed.

Occupancy grids are made from <name>_full.json by make_grid.py.
"""

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

from ifc_extract.common import write_json
from ifc_extract.extractor import IfcExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract an IFC model into a full JSON plus a compact LLM view.")
    parser.add_argument("--input", required=True, help="Path to source .ifc file")
    parser.add_argument("--output-dir", default=".", help="Directory for <name>_full.json / <name>_llm.json")
    parser.add_argument("--no-llm-view", action="store_true", help="Write only <name>_full.json")
    parser.add_argument("--drop-pset", action="append", default=[], metavar="NAME",
                        help="Exact pset/qto name to drop (repeatable), added to DROP_PSET_EXACT")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count(),
                        help="Geometry iterator threads (default: all cores)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    extractor = IfcExtractor(in_path, drop_psets=args.drop_pset, threads=args.threads)
    extractor.run()
    full = extractor.full_json()

    in_bytes = in_path.stat().st_size
    full_path = out_dir / f"{in_path.stem}_full.json"
    full_bytes = write_json(full_path, full)
    print(f"Input:  {in_path} ({in_bytes:,} bytes)")
    print(f"Output: {full_path} ({full_bytes:,} bytes, {100 * full_bytes / in_bytes:.1f}% of input)")
    if not args.no_llm_view:
        llm = extractor.llm_json(full)
        llm_path = out_dir / f"{in_path.stem}_llm.json"
        llm_bytes = write_json(llm_path, llm)
        print(f"        {llm_path} ({llm_bytes:,} bytes, {len(llm['rooms'])} rooms, "
              f"{len(llm['equipment'])} equipment)")
    extractor.audit.print_summary(extractor.model, extractor.units, full)
    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
