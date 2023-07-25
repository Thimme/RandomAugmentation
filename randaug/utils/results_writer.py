
import os
import json
from detectron2.utils.events import EventWriter
from detectron2.utils.file_io import PathManager
from collections import defaultdict


class JSONResultsWriter(EventWriter):

    def __init__(self, json_file):
        self._file_handle = PathManager.open(json_file, "a")
    
    def write(self, results):
        to_save = defaultdict(dict)
        print(results.items())
        for k, v in results.items():
            to_save[k] = v

        self._file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self._file_handle.close()