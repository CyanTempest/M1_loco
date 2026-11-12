"""Run the manual resume entry point with a persistent log and exit status."""

import argparse
import datetime
import json
import os
import signal
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", required=True, type=Path)
    args, overrides = parser.parse_known_args()
    package = Path(__file__).resolve().parent
    record = args.record_dir.resolve()
    record.mkdir(parents=True, exist_ok=True)
    status_file = record / "status.json"
    if status_file.exists():
        raise FileExistsError(f"Choose a new record directory: {record}")
    command = ["bash", str(package / "resume_mac_full.sh"), *overrides]
    state = {
        "command": command,
        "supervisor_pid": os.getpid(),
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "log": str(record / "training.log"),
    }

    def save():
        temporary = record / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(status_file)

    with (record / "training.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=package, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        state.update(status="running", training_pid=process.pid)
        save()

        def forward(signum, frame):
            del frame
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        code = process.wait()
        state.update(
            status="completed" if code == 0 else "failed_or_interrupted",
            exit_code=code,
            finished_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        save()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
