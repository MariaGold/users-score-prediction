from xmlrpc.client import Boolean
import pandas as pd
import json

class File():
    def __init__(self, file) -> None:
        self.allowed_extensions = {"json", "csv"}
        self.file = file
        self.extension = self.get_extension()
        self.path = None
        self.data = None

    def get_extension(self):
        return self.file.filename.rsplit('.', 1)[1].lower()

    def is_allowed(self) -> Boolean:
        return '.' in self.file.filename and \
           self.extension in self.allowed_extensions

    def save(self, path) -> None:
        self.path = path
        self.file.save(self.path)

    def get_data(self) -> pd.DataFrame:
        if self.extension == "json":
            self.data = pd.read_json(self.file)
        elif self.extension == "csv":
            self.data = pd.read_csv(self.file)

        return self.data

