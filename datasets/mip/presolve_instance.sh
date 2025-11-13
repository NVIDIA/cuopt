#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

for file in miplib2017/*
do
  /home/scratch.acoerduek_sw/papilo/build/bin/papilo presolve -f $file -r reduced_$file
done
