# CPU CI

Run from the repository root on Linux x86_64, Python 3.10:

```bash
bash ci/install_cpu_deps.sh
bash ci/run_cpu_tests.sh
```

`requirements.lock` pins and hashes all direct and transitive dependencies. The
PyTorch URL selects a CPU-only wheel. Regenerate deliberately with uv 0.12.7:
`uv pip compile ci/requirements.in --python-version 3.10 --generate-hashes --output-file ci/requirements.lock`.
Do not suppress install errors or retry failing tests with a weaker command.
pytest-timeout is installed; per-test thread timeout is 600 seconds and the
GitHub job timeout is 20 minutes. No-tests exit 5 is a failure, not tested behavior.

Mark GPU-only tests `pytest.mark.gpu` or `pytest.mark.cuda`; CPU CI reports their
explicit skip reasons. Missing required CPU packages or unexpected collection
errors must fail. Existing dataset/runtime smoke tests remain opt-in and their
skips appear in the summary; they are separate GPU evidence, not CPU coverage.

A green CPU job verifies only software contracts. Renderer/training/mesh changes
also require serial desktop baseline/candidate runs at exact commits, fixed
versioned inputs, splits, seeds and configs: held-out full/foreground/background
sonar errors and black baseline; surface Chamfer/Hausdorff and completeness at
stated metric tolerances; dimensions/thickness/topology; wall time/peak memory.
Explain unavailable metrics and regressions. Exploration need not beat a baseline
immediately. Other-backend review of the exact head and evidence precedes merge;
any code update needs relevant checks and renewed review. CPU-only CI changes
need failure-propagation evidence, not GPU training.

Sources: [pytest exit codes](https://docs.pytest.org/en/stable/reference/exit-codes.html),
[explicit markers](https://docs.pytest.org/en/stable/example/markers.html),
[pytest-timeout](https://github.com/pytest-dev/pytest-timeout).
