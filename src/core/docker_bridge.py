"""
Note: This module manages the sibling Docker containers.
By mounting /var/run/docker.sock, the Purple Agent can spawn 
isolated environments for any SWE-bench instance, allowing us 
to generate our own failure logs since the evaluator withholds them.
"""

import docker
import subprocess
import time
import logging

logger = logging.getLogger(__name__)

class DockerBridge:
    """Manages DooD (Docker-out-of-Docker) execution for SWE-Bench Pro."""
    
    # We added base_commit here so it can be passed in from server.py
    def __init__(self, image_name: str, base_commit: str = None, repo_dir: str = "/workspace"):
        self.image_name = image_name
        self.base_commit = base_commit 
        self.repo_dir = repo_dir
        self.client = docker.from_env()
        self.container = None

    def start_container(self) -> bool:
        """Starts the sibling container, using local cache if pulling fails."""
        try:
            logger.info(f"Attempting to pull image: {self.image_name}")
            self.client.images.pull(self.image_name)
        except Exception as e:
            try:
                self.client.images.get(self.image_name)
                logger.info(f"Image {self.image_name} found in local cache.")
            except Exception as inner_e:
                logger.error(f"Failed to find or pull image {self.image_name}: {inner_e}")
                return False

        try:
            self.container = self.client.containers.create(
                self.image_name,
                detach=True,
                entrypoint="/bin/bash",
                command=["-c", "tail -f /dev/null"],
                working_dir=self.repo_dir
            )
            self.container.start()
            logger.info(f"Started sibling container: {self.container.short_id}")
            
            # SECRET 4: The Clean Slate Checkout
            if self.base_commit:
                checkout_cmd = f"git checkout {self.base_commit} && git reset --hard {self.base_commit}"
                self.execute_command(checkout_cmd)
                
            # SECRET 7: Start background databases before the LLM takes over
            self._start_required_services()
                
            return True
            
        except Exception as e:
            logger.error(f"Container creation failed: {e}")
            return False

    def execute_command(self, command: str, timeout: int = 120) -> tuple[int, str]:
        """
        Executes a shell command. 
        SECRET 3: Uses subprocess CLI to bypass the Amber proxy EOF bug.
        """
        if not self.container:
            return -1, "Error: Container not running."

        # Wrap in container-side timeout
        docker_cmd = [
            "docker", "exec", "-w", self.repo_dir, self.container.id,
            "timeout", "-k", "5", f"{timeout}s",
            "bash", "-c", command
        ]

        t0 = time.monotonic()
        try:
            # We add a 30s grace period on the host side over the container timeout
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=timeout + 30
            )
            exit_code = result.returncode
            stdout = result.stdout.decode(errors="replace")
            stderr = result.stderr.decode(errors="replace")
            
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return 137, f"[Host subprocess timed out after {elapsed:.0f}s]"

        combined_output = stdout
        if stderr:
            combined_output = combined_output + "\n" + stderr if combined_output else stderr

        if exit_code in (124, 137):
            note = f"\n[Command timed out after {timeout}s]"
            combined_output = combined_output + note if combined_output else note.lstrip("\n")

        return exit_code, combined_output

    def stop_container(self):
        """Cleans up the sibling container."""
        if self.container:
            try:
                self.container.stop(timeout=5)
                self.container.remove(force=True)
                logger.info(f"Cleaned up container {self.container.short_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanly remove container: {e}")
            self.container = None

    def _start_required_services(self):
        """Heuristically detects and starts required databases."""
        logger.info("Scanning for required background services...")
        
        # Check for Redis
        _, out = self.execute_command("grep -qi redis package.json config.json docker-compose.y*ml 2>/dev/null && echo yes")
        if "yes" in out:
            self.execute_command("redis-server --daemonize yes --protected-mode no --appendonly yes")
            logger.info("Started Redis server.")
            
        # Check for MongoDB
        _, out = self.execute_command("grep -qi mongo package.json config.json docker-compose.y*ml 2>/dev/null && echo yes")
        if "yes" in out:
            self.execute_command("mkdir -p /data/db && mongod --fork --logpath /tmp/mongod.log --dbpath /data/db")
            logger.info("Started MongoDB.")
            
        # Check for PostgreSQL
        _, out = self.execute_command("grep -qi postgres package.json config.json docker-compose.y*ml 2>/dev/null && echo yes")
        if "yes" in out:
            self.execute_command("su - postgres -c 'pg_ctl start -D /var/lib/postgresql/data -l /tmp/pg.log' || pg_ctlcluster 14 main start")
            logger.info("Started PostgreSQL.")

    def __del__(self):
        self.stop_container()