"""
Attack Logger Wrapper
========================
"""
import logging

class Logger:
    """An abstract class for different methods of logging attack results."""

    def __init__(self, save_path = r'logger.log'):
        # Set basic logging configuration. This configures the root logger.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     Configure the logger
        logging.basicConfig(level=logging.WARNING,  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


        self.logger = logging.getLogger()

        self.filter = KeywordFilter('openai')  # Replace with the keyword you want to reject
        self.logger.addFilter(self.filter)

        self.console_handler = logging.StreamHandler()
        self.logger.addHandler(self.console_handler)

        self.file_handler = logging.FileHandler(save_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

    def log_attack_result(self, result, examples_completed=None):
        pass

    def log_summary_rows(self, rows, title, window_id):
        pass

    def log_hist(self, arr, numbins, title, window_id):
        pass

    def log_sep(self):
        pass

    def flush(self):
        pass

    def close(self):
        pass



class KeywordFilter(logging.Filter):
    def __init__(self, keyword):
        self.keyword = keyword

    def filter(self, record):
        # Check if the logged message contains the keyword
        return self.keyword not in record.getMessage()

# Create a logger

# Create and add filter
