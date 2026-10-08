#!/usr/bin/env python3
"""Launcher for ``laya-serve`` under CPython 3.14.

laya loads ModernBERT through ``transformers``, and the current ``transformers``
decorates ModernBERT's ``forward`` methods with ``@torch.compile(dynamic=True)``.
Torch refuses to even import that decorator on Python 3.14+::

    RuntimeError: torch.compile is not supported on Python 3.14+

Compiling is a pure-speed optimisation; the uncompiled graph runs identically,
just more slowly. So -- before laya (via ``transformers``) imports ModernBERT --
we swap ``torch.compile`` for a pass-through decorator and then hand off to
``laya.main()``. Everything else about laya-serve (its ``LAYA_*`` environment
variables, checkpoint selection, auth) is untouched.

The launcher only needs to run under the interpreter that has ``torch`` and
``laya`` installed (here, the project's 3.14). It is harmless elsewhere: on a
Python where ``torch.compile`` works it simply leaves the model uncompiled.

Run it like any other entry point -- ports and device come from ``LAYA_*``:

    LAYA_PORT=8000 LAYA_DEVICE=cpu python laya_runner.py
"""

from __future__ import annotations

import torch


def _passthrough_compile(*args: object, **kwargs: object):
    """Return a decorator that hands the original callable back untouched.

    Covers both ``@torch.compile`` and ``@torch.compile(dynamic=True)``: in
    either form the call is evaluated at class-body time and only has to return
    a decorator that does not alter the decorated function.
    """

    def decorator(func):
        return func

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return decorator


# Patch before importing laya: its ``build_router`` imports ``transformers``,
# which in turn imports ``modernbert`` -- the module that carries the decorator.
torch.compile = _passthrough_compile  # noqa: E501

from laya.serve import main  # noqa: E402

if __name__ == "__main__":
    main()
