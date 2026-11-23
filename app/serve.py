"""Run the API: `python -m app.serve [--port 8000]`.

On Windows, uvicorn's default Proactor event loop can't be used by psycopg's
async driver, so we drive uvicorn.Server ourselves on a Selector loop.
On Linux (Docker) plain `uvicorn app.api.main:app` works as well.
"""

import argparse

import uvicorn

from app.db import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    server = uvicorn.Server(uvicorn.Config("app.api.main:app", host=a.host, port=a.port, log_level="info"))
    run(server.serve())


if __name__ == "__main__":
    main()
