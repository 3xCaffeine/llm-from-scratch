"""
Modal VS Code Server with GPU support for LLM development.

Based on Modal's official VS Code launcher with custom GPU configuration.

Usage:
    modal run deployment/modal_vscode.py
"""

import os
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path

import modal

# Configuration
CPU = 8
MEMORY = 32768  # 32 GB
GPU = "L40S"
TIMEOUT = 3600  # 1 hour
VOLUME_NAME = "vscode-modal"
REPO_URL = "https://github.com/3xCaffeine/language-model-scratchbook.git"

# Read SSH public key for authentication
SSH_KEY_PATH = Path.home() / ".ssh" / "id_rsa.pub"

SSH_PUBLIC_KEY = SSH_KEY_PATH.read_text().strip() if SSH_KEY_PATH.exists() else None

# Code Server installation URLs
CODE_SERVER_INSTALLER = "https://code-server.dev/install.sh"
CODE_SERVER_ENTRYPOINT = "https://raw.githubusercontent.com/coder/code-server/refs/tags/v4.96.1/ci/release-image/entrypoint.sh"
FIXUID_INSTALLAER = "https://github.com/boxboat/fixuid/releases/download/v0.6.0/fixuid-0.6.0-linux-$ARCH.tar.gz"

app = modal.App("vscode-llm-scratch")

# Configure the image with CUDA support and SSH
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("curl", "dumb-init", "git", "git-lfs", "openssh-server")
    .run_commands(
        f"curl -fsSL {CODE_SERVER_INSTALLER} | sh",
        f"curl -fsSL {CODE_SERVER_ENTRYPOINT} > /code-server.sh",
        "chmod u+x /code-server.sh",
    )
    .run_commands(
        'ARCH="$(dpkg --print-architecture)"'
        f' && curl -fsSL "{FIXUID_INSTALLAER}" | tar -C /usr/local/bin -xzf - '
        " && chown root:root /usr/local/bin/fixuid"
        " && chmod 4755 /usr/local/bin/fixuid"
        " && mkdir -p /etc/fixuid"
        ' && echo "user: root" >> /etc/fixuid/config.yml'
        ' && echo "group: root" >> /etc/fixuid/config.yml'
    )
    .run_commands(
        "mkdir -p /home/coder",
        "mkdir -p /run/sshd",
        "mkdir -p /root/.ssh",
    )
    .env({"ENTRYPOINTD": ""})
)

# Create a persistent volume
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def wait_for_port(data: tuple[str, str], q: modal.Queue):
    """Wait for VS Code Server to be ready on port 8080."""
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 8080), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError(
                    "Waited too long for port 8080 to accept connections"
                ) from exc
    q.put(data)


@app.function(
    image=image,
    cpu=CPU,
    memory=MEMORY,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={"/home/coder": volume},
    max_containers=1,  # Ensure single instance for volume consistency
    secrets=[modal.Secret.from_dict({"SSH_PUBLIC_KEY": SSH_PUBLIC_KEY or ""})],
)
def run_vscode(q: modal.Queue):
    """Launch VS Code Server with GPU support and SSH access."""
    os.chdir("/home/coder")

    # Set up SSH authorized keys if provided
    ssh_key = os.environ.get("SSH_PUBLIC_KEY")
    if ssh_key:
        os.makedirs("/root/.ssh", exist_ok=True)
        with open("/root/.ssh/authorized_keys", "w") as f:
            f.write(ssh_key)
        os.chmod("/root/.ssh/authorized_keys", 0o600)

    # Clone the repository if it doesn't exist
    repo_dir = "/home/coder/language-model-scratchbook"
    if not os.path.exists(repo_dir):
        print("Cloning language-model-scratchbook repository...")
        subprocess.run(
            ["git", "clone", REPO_URL, repo_dir],
            check=True,
        )
        print(f"Repository cloned to {repo_dir}")
    else:
        print(f"Repository already exists at {repo_dir}")

    # Generate a secure token for authentication
    token = secrets.token_urlsafe(13)

    # Start SSH server in background
    subprocess.Popen(
        ["/usr/sbin/sshd", "-D", "-e"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Forward both VS Code (8080) and SSH (22) ports
    with (
        modal.forward(8080) as vscode_tunnel,
        modal.forward(22, unencrypted=True) as ssh_tunnel,
    ):
        url = vscode_tunnel.url
        ssh_host, ssh_port = ssh_tunnel.tcp_socket

        print("\nVS Code Server on Modal")
        print(f"   VS Code URL: {url}")
        print(f"   VS Code Password: {token}")
        print(f"   SSH Command: ssh -p {ssh_port} root@{ssh_host}")
        print(f"   GPU: {GPU}")
        print(f"   CPU: {CPU} cores")
        print(f"   Memory: {MEMORY} MB")
        print("   Workspace: /home/coder")
        print(f"   Repository: {repo_dir}\n")

        # Signal that the server is starting
        threading.Thread(target=wait_for_port, args=((url, token), q)).start()

        # Start code-server
        subprocess.run(
            ["/code-server.sh", "--bind-addr", "0.0.0.0:8080", "."],
            env={**os.environ, "SHELL": "/bin/bash", "PASSWORD": token},
        )

    q.put("done")


@app.local_entrypoint()
def main():
    """Local entrypoint to launch VS Code Server and open in browser."""
    print("Launching VS Code Server on Modal...")
    print(f"Configuration: {GPU} GPU, {CPU} CPUs, {MEMORY}MB RAM")
    print(f"Volume: {VOLUME_NAME}")

    if SSH_PUBLIC_KEY:
        print(f"SSH Key: {SSH_KEY_PATH}")
    else:
        print("Warning: No SSH public key found. SSH access will not be available.")
        print("Create a key with: ssh-keygen -t ed25519")
    print()

    with modal.Queue.ephemeral() as q:
        run_vscode.spawn(q)
        url, token = q.get()
        time.sleep(1)  # Give VS Code a chance to start up

        # print("Opening VS Code in your browser...")
        # webbrowser.open(url)

        print("\nUse Ctrl+C to stop the server when done.\n")
        assert q.get() == "done"
