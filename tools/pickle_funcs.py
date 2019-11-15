import pickle
from tools.log_text import log_text


def dump_data(obj, path, log_path, obj_name):
    log_text(log_path, "dumping %s to %s" % (obj_name, path))
    with open(path, "w") as writer:
        pickle.dump(obj, writer)


def load_data(path, log_path, obj_name):
    log_text(log_path, "loading data from %s to %s" % (path, obj_name))
    with open(path) as reader:
        return pickle.load(reader)