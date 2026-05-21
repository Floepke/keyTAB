import multiprocessing as mp
import sys


def preferred_start_method() -> str | None:
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        if "fork" in mp.get_all_start_methods():
            return "fork"
    return None


def configure_start_method() -> None:
    method = preferred_start_method()
    if method is None:
        return

    current_method = mp.get_start_method(allow_none=True)
    if current_method is None:
        mp.set_start_method(method)


def get_context() -> mp.context.BaseContext:
    method = preferred_start_method()
    if method is None:
        return mp.get_context()
    return mp.get_context(method)