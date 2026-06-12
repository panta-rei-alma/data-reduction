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
    # Inject the continuum product type unless the caller already set one.
    if not any(a == "--product-type" or a.startswith("--product-type=") for a in argv):
        argv = ["--product-type", "cont", *argv]
    return _dispatch_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
