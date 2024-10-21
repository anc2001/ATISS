# Use an NVIDIA CUDA runtime base image with Ubuntu
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install dependencies: OpenGL, Xvfb, and other utilities
RUN apt-get update && apt-get install -y \
    wget \
    git \
    sudo \
    build-essential \
    libgl1-mesa-glx \
    libglu1-mesa \
    mesa-utils \
    libxrender1 \
    libxext6 \
    libsm6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -tipy

# Add Conda to the PATH
ENV PATH="/opt/conda/bin:$PATH"

# Build argument to pass the host user UID
ARG USER_UID=1000

# Create a user with the specified UID
RUN useradd -m -u $USER_UID -s /bin/bash user \
    && echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create the /tmp/.X11-unix directory as root
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix && chown root:user /tmp/.X11-unix

# Switch to the new user
USER user

# Initialize Conda for the non-root user
RUN /opt/conda/bin/conda init bash

# Copy the environment.yaml file
COPY --chown=user:user environment.yml .

# Create the Conda environment using the provided environment.yaml file
RUN conda env create -f environment.yml

# Copy the project files into the container
COPY --chown=user:user . .

# Copy the project files into the container
COPY . .

# Fix ownership of the entire project directory to ensure no root-owned files remain
RUN sudo chown -R user:user /app

# Build the Python extension in-place and install the package in editable mode using conda run
RUN conda run -n atiss python setup.py build_ext --inplace
RUN conda run -n atiss pip install -e .

# Make the entrypoint script executable
COPY --chown=user:user entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint to the script
ENTRYPOINT ["/app/entrypoint.sh"]
