from NetworkIntrusionDetection.components.train_model import Training
from NetworkIntrusionDetection.config.configuration import ConfigurationManager
from NetworkIntrusionDetection import logger

STAGE_NAME = "Train Model Stage"

class TrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        training_config = config.getTrainingConfig()
        preparemodel_config = config.getPrepareModelConfig()
        obj = Training(training_config, preparemodel_config)
        obj.train_and_save_isolation_forest_model()
        obj.train_and_save_log_reg_model()
        obj.train_and_save_decision_trees_model()
        obj.train_and_save_random_forest_model()
        obj.train_and_save_xgboost_model()
        obj.train_and_save_mlp_model()
        obj.train_and_save_svm_model()
        obj.train_and_save_naive_bayes_model()
        
if __name__ == "__main__":
    try:
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e