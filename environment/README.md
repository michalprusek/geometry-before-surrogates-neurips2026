# The run environment

`requirements-frozen.txt` is the output of `uv pip freeze` for the environment that ran the camera-ready
GuacaMol campaign (114 packages). `PACKAGE_LICENCES.tsv` lists the licence each package declares.

| item | value |
|---|---|
| Python | 3.12.14 (CPython, GCC 11.5.0) |
| installer | uv 0.10.4 |
| PyTorch | 2.11.0, reported by the interpreter as `2.11.0+cu130`; CUDA 13.0, cuDNN 9.19.0 |
| BoTorch / GPyTorch / linear-operator | 0.17.2 / 1.15.2 / 0.6.1 |
| operating system | Rocky Linux 9.8, kernel 5.14.0-687.33.1.el9_8.x86_64, glibc 2.34 |
| GPU machine | 2 x NVIDIA L40S (46,068 MiB), driver 610.57.04; 2 x Intel Xeon Gold 6542Y |
| CPU-class arms (GaBO, Riemannian Matern) and LassoBench | a second machine with the same CPU model, no GPU visible; its environment holds the same 114 packages |
| threads per process | `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4` for the GuacaMol cells; one thread per LassoBench run of the camera-ready rerun (rules file, amendment log) |

Install:

```bash
uv venv --python 3.12 .venv
uv pip install --no-deps --python .venv/bin/python -r environment/requirements-frozen.txt
```

`--no-deps` installs exactly the listed packages. Without it the installation fails or grows:
checked with uv 0.10.4 for CPython 3.12 on x86_64 manylinux_2_34 (`uv pip compile`, no environment built). (1) With dependency resolution the file does not resolve: `lassobench` at the pinned commit declares `sparse-ho` as an unpinned archive URL, which conflicts with the pinned git commit of `sparse-ho`. (2) With `--no-deps` all 114 pinned distributions are found and the result equals the file. (3) With that one URL overridden the resolver keeps all 114 pins and adds 38 packages (`ax-platform`, `GPy`, `pyDOE` and their dependencies) that `lassobench` declares and the run environment did not contain.

Two packages are installed from git at fixed commits (`lassobench`, `sparse-ho`), so `git` is needed. The freeze writes `torch==2.11.0` without the local version label; on
Linux x86_64 the default wheel of that version is the CUDA 13.0 build, as the pinned `nvidia-*` packages
show.

The stored (pre-camera-ready) result files were produced in several older environments; rule section 1 of
`docs/camera_ready/RUN_RULES.md` and the baseline appendix of the paper describe them. They are not
reproduced by this environment.
