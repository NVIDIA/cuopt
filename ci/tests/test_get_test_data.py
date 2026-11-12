# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "datasets" / "get_test_data.sh"


def fake_tool(directory, name, body):
    path = directory / name
    path.write_text("#!/bin/bash\nset -eu\n" + body)
    path.chmod(0o755)


def make_tar(path, member, contents=b"new dataset"):
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo(member)
        item.size = len(contents)
        archive.addfile(item, io.BytesIO(contents))


class DatasetDownloadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bindir = self.root / "bin"
        self.bindir.mkdir()
        self.archives = self.root / "archives"
        self.archives.mkdir()
        self.workdir = self.root / "work"
        self.workdir.mkdir()
        fake_tool(
            self.bindir,
            "wget",
            """url=${!#}
name=${url##*/}
case "$FAKE_FETCH_MODE" in
  fail) exit 8 ;;
  corrupt) printf 'not an archive' > "$name" ;;
  fixtures) cp "$FAKE_ARCHIVES/$name" "$name" ;;
esac
""",
        )
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("AWS_", "CUOPT_"))
        }
        self.env.update(
            PATH=f"{self.bindir}:{self.env['PATH']}",
            FAKE_ARCHIVES=str(self.archives),
        )

    def run_script(self, option, mode):
        env = self.env | {"FAKE_FETCH_MODE": mode}
        return subprocess.run(
            ["bash", str(SCRIPT), option],
            cwd=self.workdir,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )

    def add_cvrptw_archives(self):
        for count in range(200, 1001, 200):
            with zipfile.ZipFile(
                self.archives / f"homberger_{count}_customer_instances.zip",
                "w",
            ) as archive:
                archive.writestr(f"{count}.txt", str(count))
        with zipfile.ZipFile(
            self.archives / "solomon-100.zip", "w"
        ) as archive:
            archive.writestr("solomon.txt", "solomon")

    def enable_fake_s3(self, body):
        fake_tool(self.bindir, "aws", body)
        self.env.update(
            CUOPT_S3_URI="s3://example/",
            CUOPT_AWS_ACCESS_KEY_ID="test",
            CUOPT_AWS_SECRET_ACCESS_KEY="test",
        )

    def test_http_failures_preserve_existing_data_and_tmp(self):
        (self.workdir / "cvrp").mkdir()
        (self.workdir / "cvrp" / "existing.txt").write_text("keep me")
        (self.workdir / "tmp").mkdir()
        (self.workdir / "tmp" / "unrelated.txt").write_text("also keep me")
        for mode in ("fail", "corrupt"):
            with self.subTest(mode=mode):
                result = self.run_script("--cvrp", mode)
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertEqual(
                    (self.workdir / "cvrp" / "existing.txt").read_text(),
                    "keep me",
                )
                self.assertEqual(
                    (self.workdir / "tmp" / "unrelated.txt").read_text(),
                    "also keep me",
                )

    def test_success_replaces_selected_dataset_after_extraction(self):
        make_tar(self.archives / "Vrp-Set-X.tgz", "new.txt")
        (self.workdir / "cvrp").mkdir()
        (self.workdir / "cvrp" / "old.txt").write_text("old")
        (self.workdir / "tmp").mkdir()
        (self.workdir / "tmp" / "unrelated.txt").write_text("keep me")
        result = self.run_script("--cvrp", "fixtures")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            (self.workdir / "cvrp" / "new.txt").read_text(), "new dataset"
        )
        self.assertFalse((self.workdir / "cvrp" / "old.txt").exists())
        self.assertEqual(
            (self.workdir / "tmp" / "unrelated.txt").read_text(), "keep me"
        )

    def test_multiple_archives_share_a_destination(self):
        self.add_cvrptw_archives()
        result = self.run_script("--cvrptw", "fixtures")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            sorted(path.name for path in (self.workdir / "cvrptw").iterdir()),
            ["1000.txt", "200.txt", "400.txt", "600.txt", "800.txt"],
        )
        self.assertEqual(
            (self.workdir / "solomon" / "solomon.txt").read_text(), "solomon"
        )

    def test_failed_s3_and_http_do_not_modify_existing_data(self):
        self.enable_fake_s3("""if [[ "$1 $2" == "sts get-caller-identity" ]]; then exit 0; fi
if [[ "$1 $2" == "s3 cp" ]]; then printf 'corrupt'; exit 1; fi
if [[ "$1 $2" == "s3 sync" ]]; then mkdir -p "$4"; printf 'partial' > "$4/partial.txt"; exit 1; fi
exit 1
""")
        (self.workdir / "cvrp").mkdir()
        (self.workdir / "cvrp" / "existing.txt").write_text("keep me")
        result = self.run_script("--cvrp", "fail")
        self.assertNotEqual(
            result.returncode, 0, result.stdout + result.stderr
        )
        self.assertEqual(
            (self.workdir / "cvrp" / "existing.txt").read_text(), "keep me"
        )
        self.assertFalse((self.workdir / "cvrp" / "partial.txt").exists())

    def test_successful_s3_archive_merges_without_following_destination_symlink(
        self,
    ):
        make_tar(self.archives / "s3.tar.gz", "cvrp/from_s3.txt", b"from s3")
        self.enable_fake_s3("""if [[ "$1 $2" == "sts get-caller-identity" ]]; then exit 0; fi
if [[ "$1 $2" == "s3 cp" ]]; then cat "$FAKE_ARCHIVES/s3.tar.gz"; exit 0; fi
exit 1
""")
        (self.workdir / "cvrp").mkdir()
        (self.workdir / "cvrp" / "existing.txt").write_text("keep me")
        outside = self.root / "outside.txt"
        outside.write_text("outside")
        (self.workdir / "cvrp" / "from_s3.txt").symlink_to(outside)
        result = self.run_script("--cvrp", "fail")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            (self.workdir / "cvrp" / "from_s3.txt").read_text(), "from s3"
        )
        self.assertEqual(
            (self.workdir / "cvrp" / "existing.txt").read_text(), "keep me"
        )
        self.assertEqual(outside.read_text(), "outside")

    def test_publish_failure_restores_all_original_destinations(self):
        self.add_cvrptw_archives()
        fake_tool(
            self.bindir,
            "mv",
            """if [[ "$2" == */data/solomon && "$3" == solomon ]]; then exit 1; fi
exec /usr/bin/mv "$@"
""",
        )
        for name in ("cvrptw", "solomon"):
            (self.workdir / name).mkdir()
            (self.workdir / name / "existing.txt").write_text(name)
        result = self.run_script("--cvrptw", "fixtures")
        self.assertNotEqual(
            result.returncode, 0, result.stdout + result.stderr
        )
        for name in ("cvrptw", "solomon"):
            self.assertEqual(
                (self.workdir / name / "existing.txt").read_text(), name
            )
        self.assertFalse(list(self.workdir.glob(".cuopt-datasets.*")))

    def test_s3_sync_succeeds_after_archive_failure(self):
        self.enable_fake_s3("""if [[ "$1 $2" == "sts get-caller-identity" ]]; then exit 0; fi
if [[ "$1 $2" == "s3 cp" ]]; then printf 'corrupt'; exit 1; fi
if [[ "$1 $2" == "s3 sync" ]]; then mkdir -p "$4"; printf 'synced' > "$4/from_sync.txt"; exit 0; fi
exit 1
""")
        result = self.run_script("--cvrp", "fail")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            (self.workdir / "cvrp" / "from_sync.txt").read_text(), "synced"
        )


if __name__ == "__main__":
    unittest.main()
