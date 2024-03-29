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
    db_scan_model_path: Path
    isolation_forest_model_path: Path
    lof_model_path: Path
    log_reg_model_path: Path
    decision_trees_model_path: Path
    random_forest_model_path: Path
    xgboost_model_path: Path
    svm_model_path: Path
    naive_bayes_model_path: Path
    custom_bagging_model_path: Path
    mlp_model_path: Path
    params: ConfigBox
    
# @dataclass(frozen=True)
# class TrainingConfig:
#     model_checkpoint_path: Path
#     root_dir: Path
#     trained_model_path: Path
#     updated_base_model_path: Path
#     training_data: Path
#     params_epochs: int
#     params_batch_size: int
#     params_is_augmentation: bool
#     params_image_size: list
    
# @dataclass(frozen=True)
# class EvaluationConfig:
#     path_of_model: Path
#     training_data: Path
#     all_params: dict
#     mlflow_uri: str
#     params_image_size: list
#     params_batch_size: int