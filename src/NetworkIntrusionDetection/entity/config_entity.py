from dataclasses import dataclass
from pathlib import Path
from box import ConfigBox

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class FEConfig:
    root_dir: Path
    select_k_best: int
    data_file: Path
    final_data_train: Path
    final_data_test: Path

@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    isolation_forest_model_path: Path
    log_reg_model_path: Path
    decision_trees_model_path: Path
    random_forest_model_path: Path
    xgboost_model_path: Path
    svm_model_path: Path
    naive_bayes_model_path: Path
    mlp_model_path: Path
    params: ConfigBox
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    data_train: Path
    trained_isolation_forest_model_path: Path
    trained_log_reg_model_path: Path
    trained_decision_trees_model_path: Path
    trained_random_forest_model_path: Path
    trained_xgboost_model_path: Path
    trained_svm_model_path: Path
    trained_naive_bayes_model_path: Path
    trained_mlp_model_path: Path
    
@dataclass(frozen=False)
class EvaluationConfig:
    root_dir: Path
    test_data_path: Path
    trained_isolation_forest_model_path: Path
    trained_log_reg_model_path: Path
    trained_decision_trees_model_path: Path
    trained_random_forest_model_path: Path
    trained_xgboost_model_path: Path
    trained_svm_model_path: Path
    trained_naive_bayes_model_path: Path
    trained_mlp_model_path: Path
    best_model_dir: Path
    mlflow_uri: str
    params: ConfigBox
    best_model_path: Path
    best_model_name: Path