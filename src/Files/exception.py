import sys
import traceback
from logger import CustomLogger

class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_detail or self._get_error_detail()
        self.logger = CustomLogger(log_file="logs/exceptions.log")
        self.log_exception()

    def _get_error_detail(self):
        _, _, exc_tb = sys.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error occurred in python script [{file_name}] at line number [{line_number}]"
        return "Error details unavailable"

    def log_exception(self):
        self.logger.error(f"Exception occurred: {self.error_message}")
        self.logger.error(f"Error detail: {self.error_detail}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")

    def __str__(self):
        return f"{self.error_message}\n{self.error_detail}"

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise CustomException(f"An error occurred in {func.__name__}: {str(e)}")
    return wrapper

# Usage example
if __name__ == "__main__":
    @exception_handler
    def divide_numbers(a, b):
        return a / b

    try:
        result = divide_numbers(10, 0)
    except CustomException as ce:
        print(ce)
