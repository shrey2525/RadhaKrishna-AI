from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Event:
    timestamp: float
    source: str
    event_type: str
    features: np.ndarray
    importance: Optional[float] = None
    metadata: Optional[dict] = None
