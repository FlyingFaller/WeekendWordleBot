# In src/main.py

# from weekendwordle.gui.wordle_app import run_gui
from weekendwordle.backend.cli_app import run_cli

if __name__ == "__main__":
    print("🚀 Starting Weekend Wordle...")
    run_cli()
    # run_gui()
