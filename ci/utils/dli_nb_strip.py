# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import sys

if __name__ == "__main__":
    fileName = sys.argv[1]
    print(fileName)
    with open(fileName, "r") as read_file:
        lines = read_file.readlines()

    with open(fileName, "w") as write_file:
        for line in lines:
            if "leaflet_viz" in line or "app.kernel.do_shutdown(True)" in line:
                pass
            else:
                write_file.write(line)
