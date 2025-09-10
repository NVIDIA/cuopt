# Creating configuration and data file

- For each test, create a configuration file and a corresponding data file.
- Refer `test_name_confg.json` for the format of the configuration file.
- Supported metrics can be found in `cuopt/regression/benchmark_scripts/utils.py`
- File names should start with test names followed by `config` or data depending on type of it.
- Data file should be as per openapi spec of cuopt server
- These configuration and data files needs to be uploaded to `s3://cuopt-datasets/regression_datasets/`

   ```
   aws s3 cp /path/to/files s3://cuopt-datasets/regression_datasets/
   ```
