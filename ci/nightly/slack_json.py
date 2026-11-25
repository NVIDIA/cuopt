# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON helpers for the Slack shell scripts, so they carry no inline Python.

Subcommands:
  inject --channel C [--thread-ts T] <payload>
      Print the payload JSON with ``channel`` (and ``thread_ts``) added.
  get <key> [--file PATH] [--default D]
      Print ``key`` from the JSON on stdin (or PATH). Prints D, empty by
      default, when the JSON is invalid or the key is missing.
  file-complete --file-id ID --title T --channel C [--thread-ts TS]
      [--comment TEXT]
      Print the files.completeUploadExternal payload.
"""

import argparse
import json
import sys


def cmd_inject(args):
    payload = json.loads(args.payload)
    payload["channel"] = args.channel
    if args.thread_ts:
        payload["thread_ts"] = args.thread_ts
    print(json.dumps(payload))


def cmd_get(args):
    try:
        if args.file:
            with open(args.file) as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)
        print(data.get(args.key, args.default))
    except (OSError, ValueError, AttributeError):
        print(args.default)


def cmd_file_complete(args):
    payload = {
        "files": [{"id": args.file_id, "title": args.title}],
        "channel_id": args.channel,
        "initial_comment": args.comment,
    }
    if args.thread_ts:
        payload["thread_ts"] = args.thread_ts
    print(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inject")
    p.add_argument("--channel", required=True)
    p.add_argument("--thread-ts", default="")
    p.add_argument("payload")
    p.set_defaults(func=cmd_inject)

    p = sub.add_parser("get")
    p.add_argument("key")
    p.add_argument("--file", default="")
    p.add_argument("--default", default="")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("file-complete")
    p.add_argument("--file-id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--channel", required=True)
    p.add_argument("--thread-ts", default="")
    p.add_argument("--comment", default="")
    p.set_defaults(func=cmd_file_complete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
