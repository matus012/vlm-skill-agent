"""One Gazebo topic read through the `gz` CLI, with a bounded retry of signal deaths (a recorded deviation, task 006).

On PERUN `gz topic -e -n 1 ... --json-output` died by SIGSEGV / SIGABRT in 15 of 20 M4 / M6 runs of job 143786 (never
on the local machine); one death inside the truth hook crashed M4 run 06. Rules (2026-11-05, the same code on the local machine and PERUN):
- a read killed by a signal (returncode < 0) is retried, at most MAX_RETRIES times; every death is recorded in DEATHS
  (topic, attempt, signal, stderr tail, thread, UTC) and run_task writes the ones of a run into its run.json;
- a read that still dies raises GzReadError (`gz_read_failed = True`: agent.runtime records crash_origin "gz_read", the
  scorer labels it `sim`), never a silent pass;
- any other failure (clean non-zero exit, timeout) is not retried and raises as before (CalledProcessError /
  TimeoutExpired).
STATS counts reads and CLI attempts in this process (death rate = deaths / attempts)."""
import signal
import subprocess
import threading
from datetime import datetime, timezone

MAX_RETRIES = 2
DEATHS: list[dict[str, object]] = []
STATS = {"reads": 0, "attempts": 0}


class GzReadError(subprocess.CalledProcessError):
    """A `gz topic` read killed by a signal on every one of its 1 + MAX_RETRIES attempts."""
    gz_read_failed = True


def signame(returncode: int) -> str:
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


def echo_once(topic: str, timeout: float = 20.0) -> str:
    """stdout of `gz topic -e -n 1 -t <topic> --json-output`."""
    cmd = ["gz", "topic", "-e", "-n", "1", "-t", topic, "--json-output"]
    STATS["reads"] += 1
    for attempt in range(1, MAX_RETRIES + 2):
        STATS["attempts"] += 1
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode == 0:
            return p.stdout
        if p.returncode > 0:  # a clean failure is not a CLI death: as before, no retry
            raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout, p.stderr)
        DEATHS.append({"topic": topic, "attempt": attempt, "signal": signame(p.returncode),
                       "stderr": p.stderr.strip()[-300:], "thread": threading.current_thread().name,
                       "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds")})
    raise GzReadError(p.returncode, cmd, p.stdout, p.stderr)


def since(mark: tuple[int, int, int]) -> dict[str, object]:
    """Reads, attempts and deaths after `mark` (= mark())."""
    reads, attempts, n = mark
    return {"reads": STATS["reads"] - reads, "attempts": STATS["attempts"] - attempts, "deaths": DEATHS[n:]}


def mark() -> tuple[int, int, int]:
    return STATS["reads"], STATS["attempts"], len(DEATHS)
