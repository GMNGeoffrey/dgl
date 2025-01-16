# Environment setup

```shell
python3 -m venv .venv/
source .venv/bin/activate
# Note that some of the hip pytorch APIs require an older stable version of pytorch (i.e. not a nightly build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install cython ninja
```

# Build

Make a copy of `/home/gcmn/rocm/` to your home dir `${HOME}/rocm/` to get a recent compatible rocm.

We've created a
[cmake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) file
for our common builds. You should be able to make use of it, but if you need to
do something different you can create your own CMakeUserPresets.json.

```shell
cmake --build --list-presets # Should list "rocm-6.4"
cmake --preset rocm-6.4
cmake --build --preset rocm-6.4
```

Note: to clean, run:

```shell
cmake --build --preset rocm-6.4 --target clean
```

# Test

NOTE: Remember to run `/opt/gpus.py` to acquire the lock for a GPU.

C++ unit tests.

```shell
/opt/gpus.py -n 1 out/build/rocm-6.4/runUnitTests
```

Python smoke test

```shell
/opt/gpus.py -n 1 python brium/tutorial/1_node_classification.py
```

Python unit tests for CPU (excludes distributed and graphbolt tests)

```shell
DGL_LIBRARY_PATH=out/build/rocm-6.4 \
    bash ./script/run_pytest.sh -c \
        tests/python/pytorch/ \
        --ignore=tests/python/pytorch/graphbolt \
        --ignore=tests/python/pytorch/distributed \
        --deselect=tests/python/pytorch/dataloading/test_dataloader.py::test_distributed_dataloaders

# DGL is weirdly structured in a way where you can't run both of these in the
# same command because they have conflicting test names.
DGL_LIBRARY_PATH=out/build/rocm-6.4 \
    bash ./script/run_pytest.sh -c \
        tests/python/common/ \
        --ignore=tests/python/common/test_partition.py
```

Python unit tests for GPU (excludes distributed and graphbolt tests)

```shell
DGL_LIBRARY_PATH=out/build/rocm-6.4 \
    /opt/gpus.py -n 1 \
    bash ./script/run_pytest.sh -g \
        tests/python/pytorch/ \
        --ignore=tests/python/pytorch/graphbolt \
        --ignore=tests/python/pytorch/distributed \
        --deselect=tests/python/pytorch/dataloading/test_dataloader.py::test_distributed_dataloaders

DGL_LIBRARY_PATH=out/build/rocm-6.4 \
    /opt/gpus.py -n 1 \
    bash ./script/run_pytest.sh -g \
        tests/python/common/ \
        --ignore=tests/python/common/test_partition.py
```

# Install

```shell
# Or whatever version you want to install. DGL hardcodes this path
ln -s out/build/rocm-6.4-release build
# Be sure you are under the .venv: `source .venv/bin/activate`
(cd python && python setup.py install && python setup.py build_ext --inplace)
```
