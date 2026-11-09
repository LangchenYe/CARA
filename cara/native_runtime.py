"""Build strict floating-point kernels from the shipped C sources."""
import ctypes
from hashlib import sha256
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
from time import perf_counter, process_time

_LIBRARIES = {}
_FLAGS = ("-std=c99", "-O3", "-ffp-contract=off", "-fno-fast-math")


def load_kernel(name):
    if name in _LIBRARIES:
        return _LIBRARIES[name]
    if name not in ("canonical", "fused", "fused_canonical", "saturation", "factored", "edge", "prefix_canonical", "prefix_late", "assignment"):
        raise ValueError("Unknown shipped kernel")
    source = Path(__file__).with_name("kernels") / (name + ".c")
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        raise RuntimeError("A C99 compiler is required for native paired queries")
    key = sha256(source.read_bytes() + repr((_FLAGS, compiler, platform.machine(), platform.system())).encode()).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / "cara-paired-kernels"
    cache.mkdir(exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = cache / (name + "-" + key + suffix)
    cached = library.exists()
    tick_wall, tick_cpu = perf_counter(), process_time()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    if not cached:
        fd, temporary = tempfile.mkstemp(prefix=name + "-", suffix=suffix, dir=cache)
        os.close(fd)
        try:
            linking = "-dynamiclib" if platform.system() == "Darwin" else "-shared"
            subprocess.run([compiler, *_FLAGS, "-fPIC", linking, str(source), "-lm", "-o", temporary],
                           check=True, capture_output=True, text=True)
            Path(temporary).replace(library)
        finally:
            Path(temporary).unlink(missing_ok=True)
    loaded = ctypes.CDLL(str(library))
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    metadata = {"cache_hit": cached, "build_and_load_wall_seconds": perf_counter() - tick_wall,
        "build_and_load_parent_cpu_seconds": process_time() - tick_cpu,
        "compiler_child_cpu_seconds": children_after.ru_utime + children_after.ru_stime
            - children_before.ru_utime - children_before.ru_stime,
        "compiler": compiler, "flags": list(_FLAGS), "library": str(library)}
    _LIBRARIES[name] = loaded, metadata
    return loaded, metadata
