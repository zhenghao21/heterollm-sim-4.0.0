"""Build the optional typed resource reducer with installed Cython/setuptools.

Run with the project's Python.  No dependencies are installed.  Generated C,
objects and extensions stay under tmp/resource_ledger by default.  Pass
--inplace to make the extension importable by ordinary simulator processes.
Use --module NAME to build only a selected module (repeat for several).
"""
from pathlib import Path
import argparse
import os
import sys


BUILD_MODULES = ("_resource_ledger", "_nand_pack", "_hbm_pack", "_cost_ledger",
                 "_token_program", "_physical_task", "_io_execution",
                 "_nand_execution", "_hbf_page_state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--module", action="append", choices=BUILD_MODULES,
                        help="build only this module; may be repeated")
    args = parser.parse_args()
    from Cython.Build import cythonize
    from setuptools import Extension, setup

    root = Path(__file__).resolve().parents[1]
    output = root / "tmp" / "resource_ledger"
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    flags = ["/fp:strict"] if sys.platform == "win32" else ["-fno-fast-math", "-ffp-contract=off"]
    modules = tuple(dict.fromkeys(args.module)) if args.module else BUILD_MODULES
    extensions = [Extension("heterollm_sim." + name,
                           [str(root / "src" / "heterollm_sim" / (name + ".py"))],
                           extra_compile_args=flags) for name in modules]
    build_args = ["build_ext", "--build-lib", str(output / "lib"),
                  "--build-temp", str(output / "objects")]
    if args.inplace:
        build_args.append("--inplace")
    setup(name="heterollm-resource-ledger", version="0.0.0",
          package_dir={"": "src"},
          ext_modules=cythonize(extensions, build_dir=str(output / "generated"),
                               compiler_directives={"language_level": 3}),
          script_args=build_args)


if __name__ == "__main__":
    main()
