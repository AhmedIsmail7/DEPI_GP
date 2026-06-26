#!/bin/bash
# 1. Update OS and Install System Dependencies
sudo apt update
sudo apt install nodejs ffmpeg -y

# 2. Sync Python Dependencies
# تأكد أن uv مثبت أولاً، وإلا لن يعمل هذا الأمر
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

uv sync
echo "Setup complete. Environment ready."