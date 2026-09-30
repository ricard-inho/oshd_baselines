"""Build GeoTransformer's C++ extension (``geotransformer.ext``) without nvcc.

Upstream's setup.py declares a CUDAExtension, but every source is CPU C++ (grid subsampling
and radius neighbours). The system nvcc (11.5) does not match torch's CUDA 12.1, so we compile
the same sources as a CppExtension and put the .so where ``import geotransformer.ext`` finds it.
"""
import shutil
import sys
from pathlib import Path

from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent / "third_party" / "GeoTransformer"
src = root / "geotransformer" / "extensions"
files = ["extra/cloud/cloud.cpp", "cpu/grid_subsampling/grid_subsampling.cpp",
         "cpu/grid_subsampling/grid_subsampling_cpu.cpp", "cpu/radius_neighbors/radius_neighbors.cpp",
         "cpu/radius_neighbors/radius_neighbors_cpu.cpp", "pybind.cpp"]
build = root / "build_ext"
build.mkdir(exist_ok=True)
mod = load(name="ext", sources=[str(src / f) for f in files], build_directory=str(build),
           extra_cflags=["-O3"], verbose=False)
so = Path(mod.__file__)
shutil.copy(so, root / "geotransformer" / "ext.so")
print("built", root / "geotransformer" / "ext.so", file=sys.stderr)
