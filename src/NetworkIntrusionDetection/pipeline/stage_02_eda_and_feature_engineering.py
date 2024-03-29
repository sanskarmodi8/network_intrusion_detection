from NetworkIntrusionDetection import logger
from NetworkIntrusionDetection.config.configuration import ConfigurationManager
from NetworkIntrusionDetection.components.eda_and_feature_engineering import FeatureEngineering

STAGE_NAME = "Feature Engineering Stage"

class FeatureEngineeringPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_engineering_config = config.getFEConfig()
        fe = FeatureEngineering(config=feature_engineering_config)
        fe.fe()



if __name__ == '__main__':
    try:
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
        obj = FeatureEngineeringPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e