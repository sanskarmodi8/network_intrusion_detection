from NetworkIntrusionDetection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from NetworkIntrusionDetection.pipeline.stage_02_eda_and_feature_engineering import FeatureEngineeringPipeline
from NetworkIntrusionDetection.pipeline.stage_03_prepare_model import PrepareModelPipeline
from NetworkIntrusionDetection.pipeline.stage_04_training_model import TrainingPipeline
from NetworkIntrusionDetection.pipeline.stage_05_evaluation import EvaluationPipeline
from NetworkIntrusionDetection import logger


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Feature Engineering Stage"
try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    obj = FeatureEngineeringPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Model Stage"
try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    obj = PrepareModelPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Train Model Stage"
try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    obj = TrainingPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation Stage"
try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e