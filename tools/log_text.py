import logging

def log_text(log_path, log_term):
    # log_path: the path of the log file
    # log_term: the text to be logged
    logging.basicConfig(filename=log_path, filemode="a", format=" %(message)s                | %(asctime)s ", level=logging.INFO)
    print log_term
    logging.info(log_term)