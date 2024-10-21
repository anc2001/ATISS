#!/bin/bash

# Source the non-root user's .bashrc to initialize Conda
source /home/user/.bashrc

# Start Xvfb (X virtual framebuffer) on display :99 if needed
echo "Starting Xvfb on display :99..."
export DISPLAY=:99.0
Xvfb :99 -screen 0 640x480x24 &

# If no command is provided, start an interactive bash shell
if [ -z "$1" ]; then
    exec /bin/bash
else
    # Otherwise, run the provided command
    exec "$@"
fi
