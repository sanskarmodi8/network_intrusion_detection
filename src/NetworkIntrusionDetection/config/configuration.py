from NetworkIntrusionDetection.utils.common import read_yaml, create_directories
from NetworkIntrusionDetection.entity.config_entity import (DataIngestionConfig, FEConfig)
from NetworkIntrusionDetection.constants import *

class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_directories([self.config.artifacts_root])
        
    def getDataIngestionConfig(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        
    def getFEConfig(self) -> FEConfig:
        config = self.config.eda_and_feature_engineering
        create_directories([config.root_dir])
        return FEConfig(
            root_dir=config.root_dir,
            select_k_best= self.params.select_k_best,
            data_file=config.data_file,
            final_data_train=config.final_data_train,
            final_data_test=config.final_data_test
        )

