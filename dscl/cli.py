# cli.py
# the dscl command line tool.
# currently provides one command: setup-nlp
#
# usage after installing the library:
#   dscl setup-nlp
#
# this downloads the spaCy english language model that the
# StructuralLanguageChecker needs for accurate passive voice
# and nominalization detection.
#
# if spaCy is not installed, it tells the user how to install it
# rather than crashing with an import error.

import sys
import subprocess


# ── terminal color codes ──────────────────────────────────────────────────────
# used to make the output readable at a glance.
# each code is a short escape sequence the terminal interprets as a color.

COLOR_GREEN  = "\033[92m"   # success messages
COLOR_YELLOW = "\033[93m"   # warnings and instructions
COLOR_RED    = "\033[91m"   # errors
COLOR_CYAN   = "\033[96m"   # step labels and progress
COLOR_RESET  = "\033[0m"    # back to default terminal color


def print_step(message: str) -> None:
    """Prints a step label in cyan so progress is easy to follow."""
    print(f"{COLOR_CYAN}  →  {message}{COLOR_RESET}")


def print_success(message: str) -> None:
    """Prints a success message in green."""
    print(f"{COLOR_GREEN}  ✓  {message}{COLOR_RESET}")


def print_warning(message: str) -> None:
    """Prints a warning in yellow."""
    print(f"{COLOR_YELLOW}  !  {message}{COLOR_RESET}")


def print_error(message: str) -> None:
    """Prints an error in red."""
    print(f"{COLOR_RED}  ✗  {message}{COLOR_RESET}")


def print_divider() -> None:
    """Prints a simple divider line."""
    print(f"{COLOR_CYAN}  {'─' * 52}{COLOR_RESET}")


# ── commands ──────────────────────────────────────────────────────────────────

def setup_nlp() -> None:
    """
    Downloads the spaCy english language model.

    spaCy needs a language model to analyse sentence grammar.
    This command downloads the small english model (en_core_web_sm),
    which is about 12MB and handles everything DSCL needs:
      - passive voice detection
      - nominalization detection via word root analysis
      - part-of-speech tagging

    Run this once after installing dscl[nlp].
    """
    print()
    print(f"{COLOR_CYAN}  DSCL — NLP Setup{COLOR_RESET}")
    print_divider()

    # check spaCy is installed before trying to download a model for it
    print_step("Checking spaCy is installed...")
    try:
        import spacy
        print_success(f"spaCy {spacy.__version__} found.")
    except ImportError:
        print_error("spaCy is not installed.")
        print_warning("Run this first:  pip install dscl[nlp]")
        print()
        sys.exit(1)

    # check whether the model is already downloaded
    print_step("Checking for en_core_web_sm language model...")
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print_success("en_core_web_sm is already installed.")
        print_divider()
        print_success("DSCL NLP setup is complete. No action needed.")
        print()
        return
    except OSError:
        print_warning("en_core_web_sm not found. Downloading now...")

    # download the model using spaCy's built-in download command
    print_step("Downloading en_core_web_sm (~12MB)...")
    print()

    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=False,   # show spaCy's own download progress to the user
    )

    print()
    print_divider()

    if result.returncode == 0:
        print_success("en_core_web_sm downloaded successfully.")
        print_success("DSCL NLP setup is complete.")
        print()
        print(f"  The validator will now use linguistic analysis for:")
        print(f"  • passive voice detection (all forms, not just simple past)")
        print(f"  • nominalization detection (all word forms, not just exact matches)")
        print()
    else:
        print_error("Download failed.")
        print_warning("Try running this manually:")
        print_warning("  python -m spacy download en_core_web_sm")
        print()
        sys.exit(1)


# ── entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    """
    Main entry point for the dscl command line tool.
    Called when the user types 'dscl' in their terminal.

    Usage:
        dscl setup-nlp    — download the spaCy language model
        dscl --help       — show available commands
    """
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h", "help"):
        _print_help()
        return

    command = args[0]

    if command == "setup-nlp":
        setup_nlp()
    else:
        print_error(f"Unknown command: {command}")
        print()
        _print_help()
        sys.exit(1)


def _print_help() -> None:
    """Prints available commands."""
    print()
    print(f"{COLOR_CYAN}  DSCL — Dictionary-Scoped Composition Layer{COLOR_RESET}")
    print_divider()
    print(f"  Available commands:")
    print()
    print(f"  {COLOR_CYAN}dscl setup-nlp{COLOR_RESET}")
    print(f"      Downloads the spaCy language model for accurate")
    print(f"      passive voice and nominalization checking.")
    print(f"      Run once after:  pip install dscl[nlp]")
    print()
    print(f"  {COLOR_CYAN}test_dscl.py flags{COLOR_RESET}")
    print()
    print(f"  {COLOR_CYAN}--narrative \"...\"  {COLOR_RESET}")
    print(f"      The narrative to process through DSCL.")
    print(f"      Defaults to a software licence summary if not provided.")
    print()
    print(f"  {COLOR_CYAN}--only-text{COLOR_RESET}")
    print(f"      Output plain prose only.")
    print(f"      No markdown headers, bullets, bold, tables, or lists.")
    print(f"      Use when your app or pipeline renders plain text.")
    print()
    print(f"  {COLOR_CYAN}--live{COLOR_RESET}")
    print(f"      Signal that the model has web access.")
    print(f"      Injects a currency instruction so the model uses")
    print(f"      current information for time-sensitive topics.")
    print(f"      Requires a model with web search capability.")
    print()
    print(f"  {COLOR_CYAN}--compare{COLOR_RESET}")
    print(f"      Runs the same narrative twice — once with DSCL,")
    print(f"      once without — and prints a side-by-side comparison")
    print(f"      of FK grade, sentence variation, vocabulary displacement,")
    print(f"      violations, and word count.")
    print()
    print(f"  {COLOR_CYAN}--live --only-text{COLOR_RESET}")
    print(f"      Current information, rendered as plain prose.")
    print(f"      The combination for live-feed product pipelines.")
    print()
    print(f"  {COLOR_CYAN}--live --only-text --compare{COLOR_RESET}")
    print(f"      Full diagnostic run. Constrained vs baseline,")
    print(f"      with live signal and plain prose output.")
    print()