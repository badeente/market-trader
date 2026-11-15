# AP3 training environment and compatibility evidence

## Validated stack

A separate `/home/hermes/workspace/.venvs/market-trader-ap3` was created without changing the installed AP1/AP2 workers or live report service.

Observed versions: CPython3.14.9, torch2.15.0+cpu, NumPy2.5.4. The official CPU index delivered a cp314 Linux x86_64 wheel. NumPy was installed after a first framework smoke step exposed an optional-NumPy startup warning; the subsequent step was warning-free and `pip check` passed.

A real synthetic float64 AdamW step through the proposed4-16-8-1 network passed onCPU. There are225 trainable parameters. Two fresh seed0 initializations/steps returned identical losses, and loss decreased. This validates the framework path, not market predictability or the future production trainer.

Private machine-readable evidence: `var/market-trader/acceptance/ap3-compatibility-spike.json`. No historical market model was trained by this compatibility spike.

## Runtime lock

`requirements/ap3-cpu-lock.txt` pins all11 transitive training-runtime packages with SHA-256 hashes of actually downloaded wheels. It is deliberately platform-specific to CPython3.14 / Linux x86_64 CPU. Both official PyTorchCPU and PyPI indexes are configured; accepted wheel hashes prevent substitution by a different artifact. Do not remove hashes merely to work around an unsupported platform.

Lock SHA-256 at compatibility check: `7741f62f927093d5596b808c20f18b9f3691714e259b3a311a51e594fbfe7c9a`.

A hash-checked offline resolver dry-run against the saved wheelhouse passed. Wheelhouse: `var/market-trader/releases/ap3-cpu-wheelhouse` outside Git and Web. It contains training-runtime dependencies, not the project build backend or test tools.

Example installation from the repository root into a new environment:

```sh
/home/hermes/workspace/.venvs/market-trader-review/bin/python -m venv /path/to/training-venv
/path/to/training-venv/bin/python -m pip install -r requirements/ap3-cpu-lock.txt
/path/to/training-venv/bin/python -m pip install -e . --group test
```

With the existing private wheelhouse, the first pip install can instead use:

```sh
/path/to/training-venv/bin/python -m pip install --no-index \
  --find-links /home/hermes/workspace/var/market-trader/releases/ap3-cpu-wheelhouse \
  -r requirements/ap3-cpu-lock.txt
```

The core project remains free of mandatory neural runtime dependencies. Only training imports Torch; verified JSON loading and inference work in the original core environment. All6,192 real validation predictions were reproduced there exactly without Torch or NumPy imports; its full suite passed885 tests with6 training-only skips. Test/build tools are governed separately by `pyproject.toml`/`uv.lock`; the neural runtime lock does not claim to pin every development tool.

## Reproducibility limits

Use seed0, float64, singleCPU thread and deterministic operations for the approved first model. A fixed seed alone does not promise identical results across PyTorch versions, platforms or hardware. Record actual runtime/dependency/source/template identity and verify replay artifacts on the exercised environment. Official framework reproducibility guidance: `https://docs.pytorch.org/docs/stable/notes/randomness.html`.

No privileged packages, GPU stack, public service or router/firewall changes were required.
