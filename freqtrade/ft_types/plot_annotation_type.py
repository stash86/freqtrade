from datetime import datetime
from typing import Literal

from pydantic import TypeAdapter
from typing_extensions import Required, TypedDict


class AnnotationType(TypedDict, total=False):
    type: Required[Literal["area"]]
    start: str | datetime
    end: str | datetime
    y_start: float
    y_end: float
    color: str
    label: str


AnnotationTypeTA = TypeAdapter(AnnotationType)
