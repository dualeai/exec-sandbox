"""The installed version of exec-sandbox (exec_sandbox.__version__).

A leaf module (it imports no exec_sandbox module), so the modules that need the
version import it from here and not from the package, which imports them.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("exec-sandbox")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
