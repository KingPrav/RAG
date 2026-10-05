"""Download the FastAPI docs at the pinned commit (sparse: docs/en + docs_src).

Run:  python -m docs_assistant.fetch
"""
import subprocess

from . import config


def git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def current_commit(repo) -> str | None:
    if not (repo / ".git").exists():
        return None
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    return out.stdout.strip() or None


# docs/en: the Markdown pages. docs_src: the code examples they include.
# fastapi/openapi: a few pages include snippets of FastAPI's own source.
SPARSE_PATHS = ["docs/en", "docs_src", "fastapi/openapi"]


def main():
    repo = config.REPO_DIR
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=repo)
        git("remote", "add", "origin", config.FASTAPI_REPO_URL, cwd=repo)
    git("sparse-checkout", "set", *SPARSE_PATHS, cwd=repo)   # idempotent

    if current_commit(repo) == config.FASTAPI_COMMIT:
        print(f"Already at pinned commit {config.FASTAPI_COMMIT[:10]}.")
        return

    print(f"Fetching FastAPI @ {config.FASTAPI_COMMIT[:10]} (docs only)...")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", config.FASTAPI_COMMIT, cwd=repo)
    git("checkout", "-q", "FETCH_HEAD", cwd=repo)
    print(f"Done: {repo}")


if __name__ == "__main__":
    main()
