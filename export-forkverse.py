#!/usr/bin/env python3
"""Export committed Git history for Forkverse without changing the repository.

Python 3.9+ and Git are the only requirements. By default, file contents are not
included. --include-patches includes bounded text diffs, never full file blobs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

MAX_COMMITS = 200
MAX_FILES = 5000
MAX_CHANGES = 5000
MAX_UNIQUE_PATHS = 12000
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_PATCH_BYTES = 16000
TOTAL_PATCH_BYTES = 2 * 1024 * 1024


class ExportError(Exception):
    pass


def decode(value: bytes) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExportError("仓库包含非 UTF-8 文件名或提交文字，本版导出器无法无损表示。") from error


class GitReader:
    def __init__(self, directory: Path):
        self.directory = directory
        self.env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}

    def command(self, *args: str) -> list[str]:
        return ["git", "--no-pager", "--literal-pathspecs", "-C", str(self.directory),
                "-c", "core.fsmonitor=false", "-c", "color.ui=false", *args]

    def run(self, *args: str, allow_failure: bool = False) -> bytes:
        try:
            result = subprocess.run(self.command(*args), env=self.env, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=45, check=False)
        except FileNotFoundError as error:
            raise ExportError("未找到 Git，请先安装 Git 并确保 git 命令可用。") from error
        except subprocess.TimeoutExpired as error:
            raise ExportError("Git 读取超过 45 秒。请减少 --max-commits 或选择较小的仓库。") from error
        if result.returncode and not allow_failure:
            # Do not echo arbitrary local paths, remote URLs, or subprocess output.
            raise ExportError(f"Git 读取失败（{args[0]}）。请确认路径是可读取的 Git 仓库。")
        return result.stdout if not result.returncode else b""

    def limited_patch(self, args: list[str], limit: int) -> tuple[str, bool]:
        """Stop reading large diffs early; never run external diff/textconv helpers."""
        process = subprocess.Popen(self.command(*args), env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL)
        timer = threading.Timer(45, process.kill)
        timer.daemon = True
        timer.start()
        try:
            assert process.stdout is not None
            data = process.stdout.read(limit + 1)
            truncated = len(data) > limit
            if truncated:
                process.terminate()
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise ExportError("Git diff 读取超时，请去掉 --include-patches 后重试。")
            if code and not truncated:
                raise ExportError("Git diff 读取失败，请去掉 --include-patches 后重试。")
            # Drop only an incomplete final UTF-8 character at the size boundary.
            # Other encodings cannot be displayed faithfully by this JSON format.
            bounded = data[:limit]
            try:
                text = bounded.decode("utf-8")
            except UnicodeDecodeError as error:
                if truncated and error.reason == "unexpected end of data" and error.end == len(bounded):
                    text = bounded[:error.start].decode("utf-8")
                else:
                    return "", True
            return text, truncated
        finally:
            timer.cancel()
            if process.stdout:
                process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()


def diff_args(commit: str, parents: list[str], mode: str) -> list[str]:
    args = ["diff-tree", "--no-commit-id", "--no-ext-diff", "--no-textconv", "-r", "-M", "--no-color"]
    if mode == "patch":
        args += ["-p", "--unified=3", "--src-prefix=a/", "--dst-prefix=b/"]
    else:
        args += [f"--{mode}", "-z"]
    if parents:
        args += [parents[0], commit]
    else:
        args += ["--root", commit]
    return args


def parse_changes(raw: bytes) -> list[dict[str, Any]]:
    tokens = raw.split(b"\0")
    changes: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens) and tokens[index]:
        status = decode(tokens[index])
        index += 1
        if index >= len(tokens):
            raise ExportError("Git 变更记录格式不完整。")
        first = decode(tokens[index])
        index += 1
        if status.startswith(("R", "C")):
            if index >= len(tokens):
                raise ExportError("Git 重命名记录格式不完整。")
            destination = decode(tokens[index])
            index += 1
            if status.startswith("R"):
                changes.append({"path": destination, "status": "renamed", "previousPath": first})
            else:
                changes.append({"path": destination, "status": "added"})
        else:
            kind = {"A": "added", "D": "deleted", "M": "modified", "T": "modified"}.get(status[:1])
            if not kind:
                raise ExportError("提交中包含当前版本不支持的 Git 变更状态。")
            changes.append({"path": first, "status": kind})
    return changes


def parse_numstat(raw: bytes) -> dict[str, dict[str, Any]]:
    tokens = raw.split(b"\0")
    result: dict[str, dict[str, Any]] = {}
    index = 0
    while index < len(tokens) and tokens[index]:
        entry = tokens[index]
        index += 1
        fields = entry.split(b"\t", 2)
        if len(fields) != 3:
            raise ExportError("Git 行数统计格式不完整。")
        added, deleted, name = fields
        if not name:
            if index + 1 >= len(tokens):
                raise ExportError("Git 重命名行数统计格式不完整。")
            name = tokens[index + 1]
            index += 2
        filename = decode(name)
        if added == b"-" or deleted == b"-":
            result[filename] = {"binary": True}
        else:
            result[filename] = {"additions": int(added), "deletions": int(deleted)}
    return result


def export_history(directory: Path, max_commits: int = 80, include_patches: bool = False) -> dict[str, Any]:
    if not 1 <= max_commits <= MAX_COMMITS:
        raise ExportError(f"--max-commits 必须在 1 到 {MAX_COMMITS} 之间。")
    git = GitReader(directory)
    top = git.run("rev-parse", "--show-toplevel").strip()
    if not top:
        raise ExportError("请选择非裸 Git 工作仓库的目录。")
    git.directory = Path(decode(top))
    if not git.run("rev-parse", "--verify", "HEAD", allow_failure=True).strip():
        raise ExportError("仓库还没有提交。请先创建一次提交，再导出历史。")
    current = decode(git.run("symbolic-ref", "--quiet", "--short", "HEAD", allow_failure=True)).strip()
    raw_refs = git.run("for-each-ref", "--format=%(refname:short)%00%(objectname)%00%(symref)", "refs/heads", "refs/remotes")
    refs: list[dict[str, str]] = []
    for line in raw_refs.splitlines():
        fields = line.split(b"\0")
        if len(fields) == 3 and not fields[2]:
            refs.append({"name": decode(fields[0]), "commit": decode(fields[1])})
    names = {ref["name"] for ref in refs}
    default_branch = next((name for name in ["main", "master", current, "origin/main", "origin/master"] if name in names), current or "HEAD")
    ids = [decode(value) for value in git.run("rev-list", "--all", "--topo-order", f"--max-count={max_commits}", "HEAD").splitlines()]
    ids.reverse()
    if not ids:
        raise ExportError("没有可导出的提交。")
    warnings: list[str] = []
    commits: list[dict[str, Any]] = []
    all_paths: set[str] = set()
    patch_budget = TOTAL_PATCH_BYTES
    for commit_id in ids:
        fields = git.run("show", "-s", "--format=%H%x00%P%x00%an%x00%cI%x00%B%x00", commit_id).split(b"\0", 5)
        if len(fields) < 5:
            raise ExportError("Git 提交元数据不完整。")
        parents = decode(fields[1]).split()
        files = [decode(value) for value in git.run("ls-tree", "-r", "--name-only", "--full-tree", "-z", commit_id).split(b"\0") if value]
        changes = parse_changes(git.run(*diff_args(commit_id, parents, "name-status")))
        statistics = parse_numstat(git.run(*diff_args(commit_id, parents, "numstat")))
        commit: dict[str, Any] = {
            "id": decode(fields[0]), "parents": parents,
            "author": decode(fields[2]), "date": decode(fields[3]),
            "message": decode(fields[4]).rstrip("\n"),
            "files": files[:MAX_FILES], "changes": changes[:MAX_CHANGES],
        }
        if len(files) > MAX_FILES:
            commit["filesTruncated"] = True
        if len(changes) > MAX_CHANGES:
            commit["changesTruncated"] = True
        for change in commit["changes"]:
            change.update(statistics.get(change["path"], {}))
            all_paths.add(change["path"])
            if change.get("previousPath"):
                all_paths.add(change["previousPath"])
            if include_patches and not change.get("binary"):
                if patch_budget <= 0:
                    change["patchTruncated"] = True
                    continue
                args = diff_args(commit_id, parents, "patch")
                paths = [change["path"]]
                if change.get("previousPath"):
                    paths.insert(0, change["previousPath"])
                patch, truncated = git.limited_patch([*args, "--", *paths], min(MAX_PATCH_BYTES, patch_budget))
                change["patch"] = patch
                patch_budget -= len(patch.encode("utf-8"))
                if truncated:
                    change["patchTruncated"] = True
        all_paths.update(commit["files"])
        commits.append(commit)
    selected = set(ids)
    missing = {parent for commit in commits for parent in commit["parents"] if parent not in selected}
    if missing:
        warnings.append(f"{len(missing)} 个父提交位于导出范围之外；左侧边界不代表仓库首次提交。")
    if len(refs) > 128:
        refs.sort(key=lambda ref: (ref["name"] != default_branch, ref["commit"] not in selected, ref["name"]))
        warnings.append("分支引用超过 128 个，仅导出了前 128 个引用；已选提交的父子关系不受影响。")
        refs = refs[:128]
    if any(commit.get("filesTruncated") for commit in commits):
        warnings.append(f"部分提交超过 {MAX_FILES} 个文件，快照已截断并标记。")
    if any(commit.get("changesTruncated") for commit in commits):
        warnings.append(f"部分提交超过 {MAX_CHANGES} 个文件变更，变更列表已截断并标记。")
    if any(change.get("patchTruncated") for commit in commits for change in commit["changes"]):
        warnings.append("部分 diff 已截断或省略（单文件最多 16 KB，总计最多 2 MiB）。")
    if not include_patches:
        warnings.append("此次导出未包含 diff；需要代码差异时请增加 --include-patches。")
    if len(all_paths) > MAX_UNIQUE_PATHS:
        raise ExportError(f"选中历史超过 {MAX_UNIQUE_PATHS} 个不同文件路径，请减少 --max-commits。")
    history: dict[str, Any] = {
        "format": "forkverse/v1",
        "repository": {"name": git.directory.name, "source": "local", "defaultBranch": default_branch},
        "refs": refs, "commits": commits,
    }
    if warnings:
        history["warnings"] = warnings
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description="只读导出 Git 历史，用于 Forkverse 3D 仓库演化地图。")
    parser.add_argument("--repo", default=".", help="Git 仓库目录，默认当前目录")
    parser.add_argument("--max-commits", type=int, default=80, help="最多导出的提交数，1–200，默认 80")
    parser.add_argument("--output", default="history.forkverse.json", help="输出 JSON 路径")
    parser.add_argument("--include-patches", action="store_true", help="包含有大小上限的文本 diff")
    args = parser.parse_args()
    try:
        history = export_history(Path(args.repo).expanduser().resolve(), args.max_commits, args.include_patches)
        data = (json.dumps(history, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(data) > MAX_OUTPUT_BYTES:
            raise ExportError("导出超过 8 MiB，请减少 --max-commits 或去掉 --include-patches。")
        destination = Path(args.output).expanduser()
        # Exclusive creation prevents overwriting source or a previous export by accident.
        with destination.open("xb") as output:
            output.write(data)
        print(f"已导出 {len(history['commits'])} 个提交、{len(history['refs'])} 个分支引用。")
        print(f"输出：{destination.name}（{len(data) / 1024:.1f} KiB）")
        if history.get("warnings"):
            for warning in history["warnings"]:
                print(f"范围说明：{warning}")
        return 0
    except FileExistsError:
        print("错误：输出文件已存在，请使用新的 --output 文件名。", file=sys.stderr)
        return 1
    except (ExportError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
