"""Build the compiled modules with installed Cython/setuptools.

Run with the project's Python.  No dependencies are installed.  Generated C,
objects and extensions stay under tmp/resource_ledger by default.  Pass
--inplace to make the extension importable by ordinary simulator processes.
Use --module NAME to build only a selected module (repeat for several).
The pure-Python-mode modules are optional; _hbm_fr_kernel has no Python form,
so each worktree must build it before the simulator can import.
"""
from pathlib import Path
import argparse
import os
import sys


BUILD_MODULES = ("_resource_ledger", "_nand_pack", "_hbm_pack", "_cost_ledger",
                 "_token_program", "_physical_task", "_nand_execution",
                 "_hbm_fr_kernel")
# Modules that are not pure-Python-mode sources: their files under the package.
MODULE_SOURCES = {"_hbm_fr_kernel": ("_hbm_fr_kernel.pyx", "_hbm_fr_core.c")}
MODULE_HEADERS = {"_hbm_fr_kernel": ("_hbm_fr_core.h",)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--module", action="append", choices=BUILD_MODULES,
                        help="build only this module; may be repeated")
    args = parser.parse_args()
    from Cython.Build import cythonize
    from setuptools import Extension, setup

    os.chdir(Path(__file__).resolve().parents[1])
    # Relative paths: object files mirror them, and a deep worktree exceeds MSVC's path limit otherwise.
    output = Path("tmp") / "resource_ledger"
    output.mkdir(parents=True, exist_ok=True)
    flags = ["/fp:strict"] if sys.platform == "win32" else ["-fno-fast-math", "-ffp-contract=off"]
    modules = tuple(dict.fromkeys(args.module)) if args.module else BUILD_MODULES
    package = Path("src") / "heterollm_sim"
    extensions = [Extension("heterollm_sim." + name,
                           [str(package / source) for source in MODULE_SOURCES.get(name, (name + ".py",))],
                           depends=[str(package / header) for header in MODULE_HEADERS.get(name, ())],
                           include_dirs=[str(package)],
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
