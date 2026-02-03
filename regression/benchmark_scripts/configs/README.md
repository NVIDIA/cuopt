# Creating configuration and data file for routing

- For each test, create a configuration file and a corresponding data file.
- Refer `test_name_config.json` for the format of the configuration file.
- Supported metrics can be found in `cuopt/regression/benchmark_scripts/utils.py`
- File names should start with test names followed by `config` or data depending on type of it.
- Data file should be as per openapi spec of cuopt server
- These configuration and data files needs to be uploaded to `s3://cuopt-datasets/regression_datasets/`

   ```
   aws s3 cp /path/to/files s3://cuopt-datasets/regression_datasets/
   ```

# Creating configuration and data file for lp and milp

- For each test, create a mps file
- Refer `lp_config.json` and `mip_config.json` for the format of the configuration file.
- Supported metrics can be found in `cuopt/regression/benchmark_scripts/utils.py`
- These configuration and data files needs to be in the LP_DATASETS_PATH set in config.sh
