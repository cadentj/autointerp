import os
from typing import Optional

import yaml
from pydantic import BaseModel

class ApiConfigModel(BaseModel):
    PROVIDER: Optional[str]
    APIKEY: Optional[str]

class ConfigModel(BaseModel):
    API: ApiConfigModel

    def set_provider(self, provider: str, api_key: str):

        self.API.PROVIDER = provider
        self.API.APIKEY = api_key

        self.save()

    def save(self):
        
        from .. import PATH

        with open(os.path.join(PATH, "config.yaml"), "w") as file:

            yaml.dump(self.model_dump(), file)