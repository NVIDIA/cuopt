#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get latest set of datasets
rm -rf $SCRATCH_DIR/routing_configs/*

aws s3 cp s3://cuopt-datasets/regression_datasets/ $SCRATCH_DIR/routing_configs/ --recursive

python $SCRATCH_DIR/cuopt/regression/get_datasets.py $SCRATCH_DIR/lp_datasets lp
python $SCRATCH_DIR/cuopt/regression/get_datasets.py $SCRATCH_DIR/mip_datasets mip
cp $SCRATCH_DIR/cuopt/regression/lp_config.json $SCRATCH_DIR/lp_datasets/
cp $SCRATCH_DIR/cuopt/regression/mip_config.json $SCRATCH_DIR/mip_datasets/

# Run build and test
bash $SCRATCH_DIR/cuopt/regression/cronjob.sh --benchmark  --skip-spreadsheet
