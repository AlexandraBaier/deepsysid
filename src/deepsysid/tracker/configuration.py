from typing import Optional

from pydantic import BaseModel


class BaseEventTrackerConfig(BaseModel):
    id: Optional[str] = None


class ExperimentTrackingConfiguration(BaseModel):
    tracking_class: str
    parameters: BaseEventTrackerConfig
