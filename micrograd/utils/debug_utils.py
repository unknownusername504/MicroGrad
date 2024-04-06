# Class var for debug print
class DebugPrint:
    debug = False

    def __init__(self):
        pass

    def __enter__(self):
        DebugPrint.debug = True

    def __exit__(self, exc_type, exc_value, traceback):
        DebugPrint.debug = False

    @staticmethod
    def set_debug(debug: bool):
        DebugPrint.debug = debug


def debug_print(*args, **kwargs):
    if DebugPrint.debug:
        print(*args, **kwargs)
