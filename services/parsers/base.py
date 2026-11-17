from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExtractedBlock:
    """Represents a discrete block of text extracted from a document with associated metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
