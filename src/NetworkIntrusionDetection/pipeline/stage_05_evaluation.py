from NetworkIntrusionDetection.components.evaluation import Evaluation
from NetworkIntrusionDetection.config.configuration import ConfigurationManager
from NetworkIntrusionDetection import logger
import os
from pathlib import Path
from NetworkIntrusionDetection.utils.common import read_yaml

STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        if (os.path.exists("secrets.yaml")):
            params = read_yaml(Path("params.yaml"))
            secrets = read_yaml(Path("secrets.yaml"))
            os.environ["MLFLOW_TRACKING_URI"] = params.MLFLOW_TRACKING_URI
            os.environ["MLFLOW_TRACKING_USERNAME"] = secrets.MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = secrets.MLFLOW_TRACKING_PASSWORD
            logger.info("\n\nenv var set for mlflow\n\n")
    def main(self):
        config = ConfigurationManager()
        eval_config = config.getEvaluationConfig()
        evaluation = Evaluation(eval_config)
        # evaluation.evaluate_and_log_into_mlflow()
        evaluation.evaluate_without_logging_in_mlflow()
        evaluation.save_best_model()
        
if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e