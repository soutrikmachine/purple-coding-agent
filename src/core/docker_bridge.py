import logging
import shlex
from typing import Dict, Tuple

import docker
from docker.errors import DockerException, ImageNotFound

logger = logging.getLogger("purple_agent.docker_bridge")

class DockerBridge:
    """
    Manages the sibling Docker container via the mounted /var/run/docker.sock.
    This acts as the execution engine for the LLM's Bash actions and the Test Gate.
    """

    def __init__(self, workdir: str = "/workspace/repo"):
        """
        Initializes the bridge. Relies on the host's Docker socket being mounted.
        """
        self.workdir = workdir
        self.container = None
        
        try:
            # Automatically connects to /var/run/docker.sock
            self.client = docker.from_env()
        except DockerException as e:
            logger.error("Failed to connect to Docker daemon. Is the socket mounted?")
            raise e

    def bootstrap(self, image_name: str, base_commit: str = "") -> bool:
        """
        Stage 1: Pulls the image, starts the sibling container, and checks out the commit.
        """
        logger.info(f"Bootstrapping container from image: {image_name}")
        
        try:
            # 1. Ensure image exists locally
            try:
                self.client.images.get(image_name)
            except ImageNotFound:
                logger.info(f"Image {image_name} not found locally. Pulling (this may take a while)...")
                self.client.images.pull(image_name)

            # 2. Start container in detached mode, keeping it alive
            self.container = self.client.containers.run(
                image_name,
                command="tail -f /dev/null",
                detach=True,
                auto_remove=True,  # Automatically cleans up when stopped
                working_dir=self.workdir,
                # Give the container some memory/CPU limits if needed, but defaults are usually fine for SWE-bench
            )
            logger.info(f"Container started: {self.container.short_id}")

            # 3. Checkout the base commit if provided
            if base_commit:
                logger.info(f"Checking out base commit: {base_commit[:12]}")
                exit_code, output = self.execute_bash(f"git checkout {base_commit}", timeout=60)
                
                if exit_code != 0:
                    logger.error(f"Git checkout failed: {output.get('stderr')}")
                    return False

            return True

        except Exception as e:
            logger.exception(f"Bootstrap failed: {e}")
            self.cleanup()
            return False

    def execute_bash(self, command: str, timeout: int = 120) -> Tuple[int, Dict[str, str]]:
        """
        Executes a command inside the running container safely.
        Returns: (exit_code, {"stdout": string, "stderr": string})
        """
        if not self.container:
            logger.error("Attempted to execute bash, but no container is running.")
            return -1, {"stdout": "", "stderr": "[ERROR] Container not initialized."}

        # Wrap the command using GNU timeout. 
        # This prevents the LLM from running blocking commands (e.g., starting a server without `&`)
        # and hanging the entire pipeline.
        safe_command = f"timeout {timeout} bash -c {shlex.quote(command)}"
        
        logger.debug(f"Executing: {command}")
        
        try:
            # demux=True guarantees stdout and stderr are returned as separate streams
            exit_code, streams = self.container.exec_run(
                safe_command,
                workdir=self.workdir,
                demux=True 
            )
            
            # Docker returns None for a stream if it's empty
            stdout_bytes = streams[0] if streams and streams[0] else b""
            stderr_bytes = streams[1] if streams and streams[1] else b""
            
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            # GNU timeout returns 124 if the command actually timed out
            if exit_code == 124:
                stderr = f"[ERROR] Command timed out after {timeout} seconds.\n" + stderr

            return exit_code, {"stdout": stdout, "stderr": stderr}

        except Exception as e:
            logger.exception(f"Execution error on command: {command}")
            return -1, {"stdout": "", "stderr": f"[INTERNAL ERROR] {str(e)}"}

    def write_file(self, filepath: str, content: str) -> bool:
        """
        Helper method specifically for injecting tools or patches (like ast_graph.py).
        """
        escaped_content = shlex.quote(content)
        exit_code, output = self.execute_bash(f"cat << 'EOF' > {filepath}\n{content}\nEOF")
        if exit_code != 0:
            logger.error(f"Failed to write file {filepath}: {output['stderr']}")
            return False
        return True

    def cleanup(self):
        """
        Stops the container. Since `auto_remove=True` is set, Docker will also delete it.
        """
        if self.container:
            logger.info(f"Cleaning up container {self.container.short_id}...")
            try:
                # Fast timeout to forcefully kill it quickly
                self.container.stop(timeout=2) 
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")
            finally:
                self.container = None