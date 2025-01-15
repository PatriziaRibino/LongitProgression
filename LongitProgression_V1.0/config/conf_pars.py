from dataclasses import dataclass, make_dataclass
import json

@dataclass
class Configurator:

    #parameters = Parameters()
    json_path =""
    parameters = None

    def __init__(self,json_path):
        self.json_path=json_path
        with open(self.json_path, "r") as f:
            self.parameters = json.load(f)
        self.json_config = make_dataclass("JsonConfig", self.parameters.keys())
        self.json_config = self.json_config(**self.parameters)

