from NetworkIntrusionDetection.components.evaluation import Evaluation
from NetworkIntrusionDetection.config.configuration import ConfigurationManager
from NetworkIntrusionDetection import logger

STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        eval_config = config.getEvaluationConfig()
        evaluation = Evaluation(eval_config)
        evaluation.evaluate_and_log_into_mlflow()
        # evaluation.evaluate_without_logging_in_mlflow()
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