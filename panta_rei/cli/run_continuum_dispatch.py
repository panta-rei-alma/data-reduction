"""CLI entry point for distributed 12m+7m continuum imaging.

A thin, dedicated wrapper around the shared imaging dispatcher that forces
``--product-type cont``.  Having a separate command (rather than only a flag)
means a line-imaging cron can never accidentally launch continuum, while all
the dispatch machinery is reused unchanged.

Usage::

    panta-rei-continuum-dispatch \\
        --base-dir <project-base> \\
        --output-dir <project-base>/imaging/output \\
        --machines-config <project-base>/machines.json

All other flags are identical to ``panta-rei-imaging-dispatch`` (see its
``--help``); ``--product-type`` is set automatically.
"""

from __future__ import annotations

import sys

from panta_rei.cli.run_imaging_dispatch import main as _dispatch_main


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Force the continuum product type; reject an explicit non-cont value so
    # this command can never launch line imaging.
    for i, a in enumerate(argv):
        if a == "--product-type" or a.startswith("--product-type="):
            value = a.split("=", 1)[1] if "=" in a else (
                argv[i + 1] if i + 1 < len(argv) else ""
            )
            if value != "cont":
                sys.stderr.write(
                    "panta-rei-continuum-dispatch only runs --product-type cont "
                    f"(got {value!r}); use panta-rei-imaging-dispatch for lines\n"
                )
                return 2
            break
    else:
        argv = ["--product-type", "cont", *argv]
    return _dispatch_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
