from NetworkIntrusionDetection.components.prepare_model import PrepareModel
from NetworkIntrusionDetection.config.configuration import ConfigurationManager
from NetworkIntrusionDetection import logger

STAGE_NAME = "Prepare Model Stage"

class PrepareModelPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.getPrepareModelConfig()
        prepare_model = PrepareModel(prepare_model_config)
        prepare_model.get_models()
        

if __name__ == '__main__':
    try:
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
        obj = PrepareModelPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e