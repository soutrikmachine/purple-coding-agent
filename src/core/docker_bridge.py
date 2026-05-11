"""
DockerBridge — Phase 2 v4.2.2

Changes vs. original submitted version:
  - __init__: docker.from_env() wrapped in try/except (safe error if socket missing)
  - start_container: NO network_mode or environment override — Amber manages networking,
    SWE-bench images manage their own PATH/runtimes. Original defaults are correct.
  - execute_command: stderr.strip() avoids injecting blank stderr lines into observations
  - _start_required_services: also checks requirements*.txt; postgres uses || true
  - __del__: hasattr guard prevents AttributeError if __init__ crashed mid-way

What was NOT changed (intentionally):
  - Container networking: default bridge (Amber overlay needs this)
  - Container environment: inherits from image (SWE-bench images are self-contained)
  - subprocess pattern: identical to original (SECRET 3 preserved)
"""

import docker
import subprocess
import time
import logging

logger = logging.getLogger(__name__)


class DockerBridge:
    """Manages DooD (Docker-out-of-Docker) execution for SWE-Bench Pro."""

    def __init__(self, image_name: str, base_commit: str = None, repo_dir: str = "/"):
        self.image_name  = image_name
        self.base_commit = base_commit
        self.repo_dir    = repo_dir
        self.container   = None

        try:
            self.client = docker.from_env()
        except Exception as e:
            logger.error("Docker client init failed — is /var/run/docker.sock mounted? %s", e)
            self.client = None

    def start_container(self) -> bool:
        """Starts the sibling container, using local cache if pulling fails."""
        if not self.client:
            logger.error("Docker client unavailable. Cannot start container.")
            return False

        try:
            logger.info("Attempting to pull image: %s", self.image_name)
            self.client.images.pull(self.image_name)
        except Exception as e:
            logger.warning("Pull failed (%s) — checking local cache", e)
            try:
                self.client.images.get(self.image_name)
                logger.info("Image %s found in local cache.", self.image_name)
            except Exception as inner_e:
                logger.error("Failed to find or pull image %s: %s", self.image_name, inner_e)
                return False

        try:
            # No network_mode or environment override:
            # - Amber manages networking via overlay; host mode breaks isolation
            # - SWE-bench images are self-contained with their own runtimes
            # Start with / as working dir so the container launches regardless
            # of where the repo actually is — we detect it below.
            self.container = self.client.containers.create(
                self.image_name,
                detach=True,
                entrypoint="/bin/bash",
                command=["-c", "tail -f /dev/null"],
                working_dir="/",
            )
            self.container.start()
            logger.info("Started sibling container: %s", self.container.short_id)

            # ── CRITICAL: Auto-detect actual repo location ────────────────────
            # SWE-bench images put the repo at varying paths:
            #   /testbed  (Python repos), /app  (Node/JS repos),
            #   /repo, /home/user/app, or even /  (root)
            # We cannot hardcode /workspace — it may not be the git root.
            detected = self._detect_repo_dir()
            if detected:
                self.repo_dir = detected
                logger.info("Repo detected at: %s", self.repo_dir)
            else:
                logger.warning("Could not detect repo dir — falling back to /")
                self.repo_dir = "/"

            # SECRET 7: Start background databases before the LLM takes over
            self._start_required_services()
            return True

        except Exception as e:
            logger.error("Container creation failed: %s", e)
            return False

    def execute_command(self, command: str, timeout: int = 120, workdir: str = "") -> tuple[int, str]:
        """
        Executes a shell command inside the sibling container.
        SECRET 3: Uses subprocess CLI to bypass the Amber proxy EOF bug.
        """
        if not self.container:
            return -1, "Error: Container not running."

        # Use explicit workdir if given, otherwise self.repo_dir.
        # During _detect_repo_dir, workdir="/" is passed to avoid the
        # chicken-and-egg: repo_dir defaults to "/" until detection runs.
        effective_workdir = workdir if workdir else self.repo_dir
        docker_cmd = [
            "docker", "exec", "-w", effective_workdir, self.container.id,
            "timeout", "-k", "5", f"{timeout}s",
            "bash", "-c", command,
        ]

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=timeout + 30,
            )
            exit_code = result.returncode
            stdout    = result.stdout.decode(errors="replace")
            stderr    = result.stderr.decode(errors="replace")

        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return 137, f"[Host subprocess timed out after {elapsed:.0f}s]"

        combined_output = stdout
        if stderr.strip():
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
                logger.info("Cleaned up container %s", self.container.short_id)
            except Exception as e:
                logger.warning("Failed to cleanly remove container: %s", e)
            self.container = None

    def _detect_repo_dir(self) -> str:
        """
        Auto-detect the git repository root inside the container.

        SWE-bench images vary widely in where they place the repo:
          /testbed  — most Python repos (pytest, requests, django, etc.)
          /app      — Node.js repos (NodeBB, etc.)
          /repo     — some Go repos
          /         — occasionally the repo is at the filesystem root

        Strategy:
          1. Check well-known paths first (fast, no find needed)
          2. Fall back to `find` for unusual layouts
        """
        # Try well-known SWE-bench repo locations first.
        # MUST pass workdir="/" — self.repo_dir is still "/" (default) at this
        # point and the image may not have a /workspace dir at all.
        candidates = ["/testbed", "/app", "/repo", "/home/user/app",
                      "/opt/app", "/srv", "/code", "/workspace"]
        for path in candidates:
            ec, out = self.execute_command(
                f"test -d {path}/.git && echo GIT_FOUND", timeout=5, workdir="/"
            )
            if "GIT_FOUND" in out:
                logger.info("Git repo found at known path: %s", path)
                return path

        # Fall back: find the first .git directory anywhere (depth ≤ 4)
        ec, out = self.execute_command(
            "find / -maxdepth 4 -name '.git' -type d 2>/dev/null | head -1",
            timeout=10, workdir="/"
        )
        git_dir = out.strip()
        if git_dir:
            repo = git_dir[:-5] if git_dir.endswith("/.git") else git_dir.rsplit("/.git", 1)[0]
            logger.info("Git repo found via find: %s", repo)
            return repo or "/"

        logger.warning("No .git directory found anywhere in container")
        return ""

    def _start_required_services(self):
        """Heuristically detects and starts required background databases."""
        logger.info("Scanning for required background services...")

        checks = {
            "redis": {
                "detect": "grep -rqi redis requirements*.txt package.json config.json docker-compose.y*ml 2>/dev/null",
                "start":  "redis-server --daemonize yes --protected-mode no --appendonly yes",
            },
            "mongodb": {
                "detect": "grep -rqi mongo requirements*.txt package.json config.json docker-compose.y*ml 2>/dev/null",
                "start":  "mkdir -p /data/db && mongod --fork --logpath /tmp/mongod.log --dbpath /data/db",
            },
            "postgres": {
                "detect": "grep -rqi postgres requirements*.txt package.json config.json docker-compose.y*ml 2>/dev/null",
                "start":  "su - postgres -c 'pg_ctl start -D /var/lib/postgresql/data -l /tmp/pg.log' 2>/dev/null || pg_ctlcluster 14 main start 2>/dev/null || true",
            },
        }

        for service, cfg in checks.items():
            _, out = self.execute_command(f"{cfg['detect']} && echo DETECTED", timeout=10)
            if "DETECTED" in out:
                logger.info("Detected %s dependency — starting...", service)
                self.execute_command(cfg["start"], timeout=20)
                logger.info("Started %s.", service)

    def __del__(self):
        if hasattr(self, "container") and self.container:
            self.stop_container()