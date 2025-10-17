"""
Terminal color and styling utilities for pretty console output.
"""

class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'

    # Text styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


def colorize(text, *styles):
    """Apply color/style codes to text."""
    return ''.join(styles) + str(text) + Colors.RESET


# --- Basic text functions ---
def bold(text): return colorize(text, Colors.BOLD)
def italic(text): return colorize(text, Colors.ITALIC)
def underline(text): return colorize(text, Colors.UNDERLINE)
def red(text): return colorize(text, Colors.RED)
def green(text): return colorize(text, Colors.GREEN)
def yellow(text): return colorize(text, Colors.YELLOW)
def blue(text): return colorize(text, Colors.BLUE)
def magenta(text): return colorize(text, Colors.MAGENTA)
def cyan(text): return colorize(text, Colors.CYAN)

# --- Themed wrappers ---
def success(text): return colorize(text, Colors.BOLD, Colors.GREEN)
def error(text): return colorize(text, Colors.BOLD, Colors.RED)
def warning(text): return colorize(text, Colors.BOLD, Colors.YELLOW)
def info(text): return colorize(text, Colors.CYAN)
def header(text): return colorize(text, Colors.BOLD, Colors.BLUE)


# --- Instrument color map ---
INSTRUMENT_COLORS = {
    "TESS": Colors.BRIGHT_RED,
    "GAIA": Colors.BRIGHT_BLUE,
    "ATLAS": Colors.BRIGHT_YELLOW,
    "ZTF": Colors.BRIGHT_GREEN,
    "BLACKGEM": Colors.BRIGHT_MAGENTA,
}


def format_prefix(gaia_id=None, instrument=None):
    """Format the message prefix with GAIA ID and instrument."""
    prefix_parts = []
    if gaia_id is not None:
        prefix_parts.append(str(gaia_id))
    if instrument is not None:
        color = INSTRUMENT_COLORS.get(instrument.upper(), Colors.WHITE)
        prefix_parts.append(colorize(instrument.upper(), color, Colors.BOLD))
    if prefix_parts:
        return "[" + "][".join(prefix_parts) + "] "
    return ""


# --- Printing functions ---
def print_success(message, gaia_id=None, instrument=None):
    print(f"{format_prefix(gaia_id, instrument)}{success(message)}")


def print_error(message, gaia_id=None, instrument=None):
    print(f"{format_prefix(gaia_id, instrument)}{error(message)}")


def print_warning(message, gaia_id=None, instrument=None):
    print(f"{format_prefix(gaia_id, instrument)}{warning(message)}")


def print_info(message, gaia_id=None, instrument=None):
    print(f"{format_prefix(gaia_id, instrument)}{info(message)}")


def print_header(message):
    print(f"\n{header('═' * 60)}")
    print(f"{header(message.center(60))}")
    print(f"{header('═' * 60)}\n")


def print_box(message, color_fn=None):
    lines = message.split('\n')
    max_len = max(len(line) for line in lines)
    border = '─' * (max_len + 2)
    if color_fn:
        print(color_fn(f"┌{border}┐"))
        for line in lines:
            print(color_fn(f"│ {line.ljust(max_len)} │"))
        print(color_fn(f"└{border}┘"))
    else:
        print(f"┌{border}┐")
        for line in lines:
            print(f"│ {line.ljust(max_len)} │")
        print(f"└{border}┘")
