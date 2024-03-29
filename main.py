from NetworkIntrusionDetection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
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

