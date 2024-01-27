DEBUG = True


# Simple function to control debug printing
def debug_print(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)
    else:
        pass
