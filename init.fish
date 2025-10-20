#!/usr/bin/env fish

if not test -d .venv
    echo "Virtual environment (.venv) not found. Creating it now..."
    uv venv --system-site-packages .venv
    source .venv/bin/activate.fish
    echo "Installing packages..."
    uv pip install python-lsp-server[all] ruff mypy
    echo "Packages installed successfully."
else
    echo "Virtual environment (.venv) found. Activating it..."
    source .venv/bin/activate.fish
    echo "Virtual environment activated."
end
