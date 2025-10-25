from __future__ import annotations

import os
import pathlib
import platform
import subprocess
import sys
from typing import List

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = pathlib.Path(sourcedir).resolve()


class CMakeBuildExt(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        extdir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        cfg = "Debug" if self.debug else "Release"
        cmake_args: List[str] = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if "CXXFLAGS" not in env:
            env["CXXFLAGS"] = ""
        env["CXXFLAGS"] += f" -DVERSION_INFO=\\\"{self.distribution.get_version()}\\\""

        python_arch = platform.architecture()[0]
        if python_arch == "32bit":
            cmake_args.append("-A")
            cmake_args.append("Win32")

        subprocess.check_call(
            ["cmake", str(ext.sourcedir), *cmake_args],
            cwd=build_temp,
            env=env,
        )

        build_args = ["--config", cfg]
        if self.parallel:
            build_args.extend(["-j", str(self.parallel)])
        subprocess.check_call(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
        )


setup(
    ext_modules=[CMakeExtension("binstatcuda._core", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuildExt},
)
