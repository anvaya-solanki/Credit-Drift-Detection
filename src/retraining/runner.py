import subprocess
from pathlib import Path

def run_retraining(data_path: Path):
    """
    Model retraining using retraining dataset.
    """
    command = [
        "python",
        "-m",
        "src.models.train",
        "--data",
        str(data_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr
    }
