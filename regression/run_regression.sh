# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Get latest set of datasets
rm -rf $SCRATCH_DIR/routing_configs/*
rm -rf $SCRATCH_DIR/lp_configs/*
rm -rf $SCRATCH_DIR/mip_configs/*

aws s3 cp s3://cuopt-datasets/regression_datasets/ $SCRATCH_DIR/routing_configs/ --recursive
aws s3 cp s3://cuopt-datasets/lp_datasets/ $SCRATCH_DIR/lp_configs/ --recursive
aws s3 cp s3://cuopt-datasets/mip_datasets/ $SCRATCH_DIR/mip_configs/ --recursive

bash $SCRATCH_DIR/cuopt/regression/get_datasets.sh

# Run build and test
bash $SCRATCH_DIR/cuopt/regression/cronjob.sh --benchmark  --skip-spreadsheet
