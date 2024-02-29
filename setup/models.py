from pydantic import BaseModel, validator, field_validator, model_validator
from typing import Union, Optional


class DimensionConfiguration(BaseModel):
    """
    This class is used to define the boundaries of the doe.
    It can also contain information regarding the optimization boundaries, if necessary.
    The optimization boundaries are by default set to the same as the doe boundaries.

    Attributes:
            name: variable name.
            min: minimum value in doe.
            max: maximum value in doe.
            integers_only: boolean operator deciding if this variable needs to be an integer.
            optimization_min: independent minimum value for optimization of this variable. Defaults to "min".
            optimization_max: independent maximum value used for optimization of this variable. Defaults to "max".
    """
    name: str
    min: Union[float, int]
    max: Union[float, int]
    integers_only: bool = False
    optimization_min: Optional[Union[float, int]] = None
    optimization_max: Optional[Union[float, int]] = None

    @model_validator(mode="after")
    def set_default_optimization_bounds(self):
        if self.optimization_min is None:
            self.optimization_min = self.min
        if self.optimization_max is None:
            self.optimization_max = self.max

        return self
