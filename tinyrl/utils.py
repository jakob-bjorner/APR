import subprocess
import time
import requests
from typing import Optional
from sglang.srt.utils import kill_process_tree

def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
    model_name: str = "model",
    api_key: Optional[str] = None,
    other_args: list[str] = (),
    env: Optional[dict] = None,
    return_stdout_stderr: Optional[tuple] = None,
):
    _, host, port = base_url.split(":")
    host = host[2:]

    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--served-model-name",
        model_name,
        *other_args,
    ]

    if api_key:
        command += ["--api-key", api_key]

    if return_stdout_stderr:
        process = subprocess.Popen(
            command,
            stdout=return_stdout_stderr[0],
            stderr=return_stdout_stderr[1],
            env=env,
            text=True,
        )
    else:
        process = subprocess.Popen(
            command, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            env=env
        )

    start_time = time.time()
    with requests.Session() as session:
        while time.time() - start_time < timeout:
            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {api_key}",
                }
                response = session.get(
                    f"{base_url}/health_generate",
                    headers=headers,
                )
                if response.status_code == 200:
                    return process
            except requests.RequestException:
                pass
            time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")

def terminate_process(process):
    kill_process_tree(process.pid)
