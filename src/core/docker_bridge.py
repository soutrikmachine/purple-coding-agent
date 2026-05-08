"""
Note: This module manages the sibling Docker containers.
By mounting /var/run/docker.sock, the Purple Agent can spawn 
isolated environments for any SWE-bench instance, allowing us 
to generate our own failure logs since the evaluator withholds them.
"""

import docker
import logging
import time
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DockerBridge:
    """
    Manages the sibling Docker containers.
    Mounts /var/run/docker.sock to spawn isolated environments for SWE-bench instances.
    """
    def __init__(self, image_name: str, container_name: Optional[str] = None):
        try:
            self.client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to connect to Docker socket: {e}")
            raise
            
        self.image_name = image_name
        self.container_name = container_name or f"purple-exec-{int(time.time())}"
        self.container = None

    def start_container(self) -> bool:
        logger.info(f"Starting container: {self.container_name} from {self.image_name}")
        try:
            self.container = self.client.containers.run(
                self.image_name,
                name=self.container_name,
                detach=True,
                tty=True,
                stdin_open=True,
                working_dir="/workspace",
                mem_limit="4g",
                network_mode="bridge"
            )
            return True
        except Exception as e:
            logger.error(f"Container startup failed: {e}")
            return False

    def execute_command(self, command: str, timeout: int = 60) -> Tuple[int, str]:
        if not self.container:
            return 1, "Error: Container not started."

        # Shell-level timeout to prevent infinite loops (e.g., hanging grep)
        safe_command = f"timeout {timeout}s bash -c {docker.utils.quote_executable(command)}"
        
        try:
            # demux=True ensures STDOUT and STDERR are cleanly separated
            exit_code, output = self.container.exec_run(
                cmd=["bash", "-c", command],
                demux=True
            )
            
            stdout, stderr = output
            combined_output = ""
            if stdout:
                combined_output += stdout.decode('utf-8', errors='replace')
            if stderr:
                combined_output += f"\nSTDERR:\n{stderr.decode('utf-8', errors='replace')}"
                
            return exit_code, combined_output.strip()
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return 1, str(e)

    def stop_container(self):
        if self.container:
            logger.info(f"Stopping and removing container: {self.container_name}")
            try:
                self.container.stop()
                self.container.remove()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

    def __del__(self):
        if hasattr(self, 'container') and self.container:
            try:
                self.container.remove(force=True)
            except:
                pass