#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import os
from datetime import datetime

class Gsheet_Report:
    def __init__(self, results_dir):
        self.benchmark_dir_path = Path(results_dir)/"benchmarks"
        self.benchmark_result_list = list(self.benchmark_dir_path.glob("benchmark_result*"))
        # FIXME: This is a default list of the current MNMG algos benchmarkee, this is subject to change
        self.map_algo_sheet = {'bfs':"BFS", "sssp":"SSSP", "louvain":"Louvain", "pagerank":"Pagerank", "wcc":"WCC", "katz":"Katz"}
        self.algos = None
        self.sheet_names = None
        self.spreadsheet = None
        self.gc = None
    
    def _setup_authentication(self):
        # Setup authentication and open the spreasheet
        # Before running cronjob, run a script setting the credential path
        if os.environ.get("GOOGLE_SHEETS_CREDENTIALS_PATH", None):
            credentials_path = os.environ["GOOGLE_SHEETS_CREDENTIALS_PATH"]
            self.gc = gspread.service_account(filename=credentials_path)
        else:
            raise Exception("Invalid credentials path")
        
    def _import_sample_worksheet(self):
        # import a sample benchmark result table and copy it to the new benchmark result spreadsheet
        date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        spreadsheet_name = f"MNMG-benchmark-results {date_time}"
        self.spreadsheet = self.gc.create(spreadsheet_name)
        
        sample_spreadsheet_name = "sample"
        
        def from_sample_spreadsheet(gc, sample_spreadsheet_name, spreadsheet):
            sample_spreadsheet = self.gc.open(sample_spreadsheet_name)
            sample_worksheet = sample_spreadsheet.worksheet('sample')
            sample_worksheet.copy_to(spreadsheet.id)
        
        from_sample_spreadsheet(self.gc, sample_spreadsheet_name, self.spreadsheet)
        
        # The new create spreadsheet has a default worksheet
        # delete that worksheet
        self.spreadsheet.del_worksheet(self.spreadsheet.get_worksheet(0))
        # Rename the only worksheet to sheet to sample
        # This worksheet containing an empty result table will be copied to as
        # many sheet as there algos in the benchmark result dir
        self.spreadsheet.get_worksheet(0).update_title('sample')
        
        # Send the new create spreadsheet to my google drive
        self.spreadsheet.share('jnke2016@gmail.com', perm_type='user', role='writer')
    
    def _extract_sheet_names(self):
        # From the benchmark result dir, get the list of algo's name that were run
        # Those will be used to as worksheet's name 

        # if the benchmark result list is empty, no benchmark were run
        if len(self.benchmark_result_list) == 0:
            return

        algos = map(lambda x: str(x).split('.')[1].split('.')[0], list(self.benchmark_result_list))
        # remove duplicates
        self.algos = list(set(algos))
        self.sheet_names = map(lambda x:self.map_algo_sheet[x], self.algos)
        return True

    
    def _create_worksheets(self, sheet_names=None):
        # Create as many worksheet as there are algos in the benchmark results dir
        if not isinstance(sheet_names, list) and sheet_names is not None:
            sheet_names = [sheet_names]

        if sheet_names is not None:
            valid_algos_benchmarked = set(self.algos) & set(sheet_names)
            # Do not create the spreadsheet of an algo which wasn't benchmarked 
            if len(valid_algos_benchmarked) < len(sheet_names):
                raise Exception(f"Invalid algo(s) specified: \n"
                            "The list of algos benchmarked are "f"{self.algos}")
                   
        worksheet = self.spreadsheet.worksheet('sample')
        # If no sheet names provided, create worksheets for all MNMG algos in 
        if sheet_names is None:
            sheet_names = self.sheet_names
     
        for sheet_name in sheet_names:
            if sheet_name in self.map_algo_sheet.keys():
                self.spreadsheet.duplicate_sheet(source_sheet_id=worksheet.id, new_sheet_name=self.map_algo_sheet[sheet_name])  
            else:
                self.spreadsheet.duplicate_sheet(source_sheet_id=worksheet.id, new_sheet_name=sheet_name)  
    
    def _write_gsheet(self, algos=None):
        # Write the results from the json to the corresponding cell in the worksheet
        def extract_cell(spreadsheet, sheet_name, scale, ngpus):
            worksheet = spreadsheet.worksheet(sheet_name)
            # The row containing the number of GPUs is 'ngpus_row'+1
            ngpus_row = worksheet.find("Number of GPUs").row
            # Find the number of GPUs cell in that row
            ngpus_algo_col = worksheet.find(str(ngpus), in_row=ngpus_row+1).col
            # Get the column containing the scale
            scale_col = worksheet.find("Scale").col
            # Find the scale's row within 'scale_col'
            scale_algo_row = worksheet.find(str(scale), in_column=scale_col).row
            return worksheet, scale_algo_row , ngpus_algo_col

        if algos is not None:
            if not isinstance(algos, list):
                algos = [algos]
            # ensure the algos specified were benchmarked
            valid_algos_benchmarked = set(self.algos) & set(algos)
            if len(valid_algos_benchmarked) == 0:
                raise Exception("Invalid algo(s) specified:\n"
                             f"{algos}" " not a subset of " f"{self.algos}")
            benchmark_result_list=[]
            # Get a list of the json files that will be scan to update the spreadsheet
            for file_name in self.benchmark_result_list:
                algo_file = str(file_name).split('.')[1].split('.')[0]
                # Only create/update spreadsheet of the algos specified
                if algo_file in algos:
                    benchmark_result_list.append(file_name)
            self.benchmark_result_list = benchmark_result_list
        for file_name in self.benchmark_result_list:
            time.sleep(5)
            with open(file_name, 'r') as openfile:
                bResult_dic = json.load(openfile)
                sheet_name = bResult_dic["funcName"].split('.')[1]
                scale = bResult_dic["argNameValuePairs"][0][1]
                ngpus = bResult_dic["argNameValuePairs"][1][1]
                result = bResult_dic["result"]
                worksheet, row, col = extract_cell(self.spreadsheet, self.map_algo_sheet[sheet_name], scale, ngpus)
                worksheet.update_cell(row, col, result)
        
        # delete sample worksheet
        self.spreadsheet.del_worksheet(self.spreadsheet.worksheet("sample"))
    
    def _get_spreadsheet_url(self):
        url_prefix = "https://docs.google.com/spreadsheets/d/"
        spreadsheet_url = f"{url_prefix}{self.spreadsheet.id}"
        print("spreadsheet url is", spreadsheet_url)
        

    def update_spreadsheet(self, algos=None):
        self._setup_authentication()
        self._import_sample_worksheet()
        self._get_spreadsheet_url()
        benchmark_json = self._extract_sheet_names()
        # Only proceed if there are benchmark results
        if benchmark_json :
            self._create_worksheets(algos)
            self._write_gsheet(algos)
    

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, required=True,
                    help="directory to store the results in json files")
    args = ap.parse_args()
    
    gsheet_report = Gsheet_Report(results_dir=args.results_dir)
    gsheet_report.update_spreadsheet()

