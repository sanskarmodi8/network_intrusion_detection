from NetworkIntrusionDetection.utils.common import read_yaml, create_directories
from NetworkIntrusionDetection.entity.config_entity import (DataIngestionConfig, FEConfig, PrepareModelConfig, TrainingConfig)
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
        
    def getPrepareModelConfig(self) -> PrepareModelConfig:
        config = self.config.prepare_model
        create_directories([config.root_dir])
        return PrepareModelConfig(
            root_dir=config.root_dir,
            db_scan_model_path=config.db_scan_model_path,
            isolation_forest_model_path=config.isolation_forest_model_path,
            lof_model_path=config.lof_model_path,
            log_reg_model_path=config.log_reg_model_path,
            decision_trees_model_path=config.decision_trees_model_path,
            random_forest_model_path=config.random_forest_model_path,
            xgboost_model_path=config.xgboost_model_path,
            svm_model_path=config.svm_model_path,
            naive_bayes_model_path=config.naive_bayes_model_path,
            mlp_model_path=config.mlp_model_path,
            params=self.params
        )
        
    def getTrainingConfig(self) -> TrainingConfig:
        config = self.config.training
        create_directories([config.root_dir])
        return TrainingConfig(
            root_dir=config.root_dir,
            data_train=config.train_data_path,
            trained_db_scan_model_path=config.trained_db_scan_model_path,
            trained_isolation_forest_model_path=config.trained_isolation_forest_model_path,
            trained_lof_model_path=config.trained_lof_model_path,
            trained_log_reg_model_path=config.trained_log_reg_model_path,
            trained_decision_trees_model_path=config.trained_decision_trees_model_path,
            trained_random_forest_model_path=config.trained_random_forest_model_path,
            trained_xgboost_model_path=config.trained_xgboost_model_path,
            trained_svm_model_path=config.trained_svm_model_path,
            trained_naive_bayes_model_path=config.trained_naive_bayes_model_path,
            trained_mlp_model_path=config.trained_mlp_model_path
        )

