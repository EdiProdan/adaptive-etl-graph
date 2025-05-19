"""
Main pipeline controller that orchestrates the entire data processing flow
"""
from src.ingestion.input_handler import InputHandler
from src.pipeline.text_pipeline import TextPipeline
from src.pipeline.image_pipeline import ImagePipeline


class PipelineController:
    """
    Controls the overall processing pipeline for multimodal data
    """

    def __init__(self, config):
        """
        Initialize the pipeline controller

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.input_handler = InputHandler(config.get('input', {}))
        self.text_pipeline = TextPipeline(config.get('text_pipeline', {}))
        self.image_pipeline = ImagePipeline(config.get('image_pipeline', {}))

    def process(self, input_source):
        """
        Process an input source through the appropriate pipeline

        Args:
            input_source: Path to file or raw content

        Returns:
            Processed data result
        """
        try:
            # Detect input type and prepare data
            input_data = self.input_handler.process_input(input_source)

            # Route to appropriate pipeline
            if input_data['type'] == 'text':
                return self.text_pipeline.process(input_data)
            elif input_data['type'] == 'image':
                return self.image_pipeline.process(input_data)
            else:
                raise ValueError(f"Unsupported input type: {input_data['type']}")

        except Exception as e:
            # Log error and return failure
            print(f"Error processing {input_source}: {str(e)}")
            return {'status': 'error', 'message': str(e)}