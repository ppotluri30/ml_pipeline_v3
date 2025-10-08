import pandas as pd
import requests
import json
import logging
from typing import Optional
from datetime import datetime
import os
class DruidIngester:
    def __init__(self, coordinator_url=None):
        if coordinator_url is None:
            coordinator_url = os.getenv("DRUID_URL", "http://coordinator:8081")
        self.coordinator_url = coordinator_url.rstrip("/")
        self.logger = logging.getLogger(__name__)
    
    def ingest_dataframe(self, df: pd.DataFrame, datasource_name: str, timestamp_column: Optional[str], overwrite: bool=True):
        """
        Ingest a pandas DataFrame into Druid
        
        Args:
            df: DataFrame to ingest
            datasource_name: Name of the Druid datasource
            timestamp_column: Column to use as timestamp
            overwrite: If True, replace existing data (default: True)
        """
        try:
            # Prepare the data
            df_copy = df.copy()
            
            print(df.columns)
            print(df.head(5))

            # Handle timestamp column
            if not (timestamp_column and timestamp_column in df.columns):
                print("No timestamp column given, assuming index represents time")
                try:
                    df_copy.reset_index(names="time")
                    timestamp_column = "time"
                except Exception as e:
                    self.logger.error(f"Error parsing time: {str(e)}")
                    print(f"❌ Error parsing time: {str(e)}")

            df_copy['__time'] = pd.to_datetime(df_copy[timestamp_column]).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            if timestamp_column != '__time':
                df_copy = df_copy.drop(columns=[timestamp_column])
            
            # Convert NaN to None
            df_copy = df_copy.where(pd.notnull(df_copy), None)
            
            # Convert to records
            records = df_copy.to_dict('records')
            
            # Get time range for interval specification
            timestamps = pd.to_datetime([record['__time'] for record in records])
            min_time = timestamps.min().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            max_time = (timestamps.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            print(f"Min time: {min_time}, Max time: {max_time}")

            
            # Create ingestion spec with overwrite support
            ingestion_spec = self._create_ingestion_spec(
                records, datasource_name, df_copy.dtypes, 
                overwrite=overwrite, interval=f"{min_time}/{max_time}"
            )
            
            # Submit to Druid
            response = requests.post(
                f"{self.coordinator_url}/druid/indexer/v1/task",
                json=ingestion_spec,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                task_id = response.json().get('task')
                self.logger.info(f"Successfully submitted ingestion task: {task_id}")
                print(f"✅ Task submitted: {task_id}")
                return task_id
            else:
                self.logger.error(f"Failed to ingest data: {response.text}")
                print(f"❌ Failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error ingesting dataframe: {str(e)}")
            print(f"❌ Error ingesting dataframe: {str(e)}")
            return None
    
    def _create_ingestion_spec(self, records, datasource_name, dtypes, overwrite=True, interval=None):
        """
        Create a Druid ingestion specification with overwrite support
        """
        # Convert records to newline-delimited JSON
        data_lines = []
        for record in records:
            try:
                data_lines.append(json.dumps(record, default=str))
            except Exception as e:
                print(f"Error serializing record: {record}, Error: {e}")
                continue
        
        data_json = "\n".join(data_lines)
        
        # Only create dimensions - no automatic sum metrics
        dimensions = []
        
        for col_name, dtype in dtypes.items():
            if col_name == '__time':
                continue
                
            if pd.api.types.is_numeric_dtype(dtype):
                dimensions.append({
                    "name": col_name,
                    "type": "double" if pd.api.types.is_float_dtype(dtype) else "long"
                })
            else:
                dimensions.append({
                    "name": col_name,
                    "type": "string"
                })
        
        # Only add count metric (no sum columns)
        metrics = [{
            "type": "count",
            "name": "count"
        }]
        
        # Base ingestion spec
        ingestion_spec = {
            "type": "index_parallel",
            "spec": {
                "ioConfig": {
                    "type": "index_parallel",
                    "inputSource": {
                        "type": "inline",
                        "data": data_json
                    },
                    "inputFormat": {
                        "type": "json"
                    },
                    "appendToExisting": not overwrite
                },
                "tuningConfig": {
                    "type": "index_parallel",
                    "maxRowsPerSegment": 5000000,
                    "maxRowsInMemory": 1000000,
                    "forceGuaranteedRollup": True
                },
                "dataSchema": {
                    "dataSource": datasource_name,
                    "timestampSpec": {
                        "column": "__time",
                        "format": "auto"
                    },
                    "dimensionsSpec": {
                        "dimensions": dimensions
                    },
                    "metricsSpec": metrics,  # Only count metric
                    "granularitySpec": {
                        "type": "uniform",
                        "segmentGranularity": "day",
                        "queryGranularity": "none",
                        "rollup": True,
                        "intervals": [interval] if interval and overwrite else None
                    }
                }
            }
        }
        
        # Remove None values from granularitySpec
        if ingestion_spec["spec"]["dataSchema"]["granularitySpec"]["intervals"] is None:
            del ingestion_spec["spec"]["dataSchema"]["granularitySpec"]["intervals"]
        
        return ingestion_spec
    
    def drop_datasource(self, datasource_name):
        """
        Completely drop/delete a datasource and all its data
        """
        try:
            response = requests.delete(
                f"{self.coordinator_url}/druid/coordinator/v1/datasources/{datasource_name}"
            )
            
            if response.status_code in [200, 202]:
                print(f"✅ Datasource {datasource_name} marked for deletion")
                return True
            else:
                print(f"❌ Failed to delete datasource: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting datasource: {e}")
            return False
    
    def check_task_status(self, task_id):
        """
        Check the status of an ingestion task
        """
        try:
            response = requests.get(f"{self.coordinator_url}/druid/indexer/v1/task/{task_id}/status")
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error checking task status: {str(e)}")
            return None