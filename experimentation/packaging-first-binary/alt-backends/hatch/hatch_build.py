"""A plain custom build hook: compile the shim ourselves, ship it as a file.
Unix only as written; MSVC would need its own branch (cl.exe /LD, .def or
__declspec exports) that this file does not have."""
import os, subprocess, sysconfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class NashimHook(BuildHookInterface):
    def initialize(self, version, build_data):
        pkg = os.path.join(self.root, "src", "d2shim_hatch")
        out = os.path.join(pkg, "libnashim" + (sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"))
        cc = os.environ.get("CC", "cc")
        subprocess.check_call([cc, "-O2", "-fPIC", "-shared", "-I" + os.path.join(pkg, "vendor"),
                               "-o", out, os.path.join(pkg, "c", "nashim.c"),
                               os.path.join(pkg, "vendor", "nanoarrow.c")])
        build_data["pure_python"] = False
        build_data["infer_tag"] = True          # -> cp3XY-cp3XY-<plat>; py3-none needs a retag
        build_data["force_include"][out] = "d2shim_hatch/" + os.path.basename(out)
