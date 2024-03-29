from NetworkIntrusionDetection.config.configuration import ConfigurationManager
from NetworkIntrusionDetection.components.data_ingestion import DataIngestion
from NetworkIntrusionDetection import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.getDataIngestionConfig()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()



if __name__ == '__main__':
    try:
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e