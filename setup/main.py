import math

from pyDOE import lhs
from models import *
import pandas as pd

N = 500     # Sample size
sampling_criterion = 'center'    # None = randomize within stratum, center will ensure uniformity
design_config = [
    DimensionConfiguration(name='VANE_TOTAL_COUNT', min=8, max=18, integers_only=True),
    DimensionConfiguration(name='VANE_LEAN', min=0, max=20),
    DimensionConfiguration(name='T_VANE_REG', min=1.4, max=3),
    DimensionConfiguration(name='T_VANE_MNT', min=1.4, max=3),
    DimensionConfiguration(name='T_HUB_REG', min=1.4, max=3),
    DimensionConfiguration(name='T_HUB_MNT', min=1.4, max=3),
    DimensionConfiguration(name='T_OUTER_REG', min=1.4, max=3),
    DimensionConfiguration(name='T_OUTER_MNT', min=1.4, max=3),
]

design = lhs(len(design_config), samples=N, criterion=sampling_criterion)

for row in design:
    for i in range(0, len(row)):
        dim_conf = design_config[i]
        min = dim_conf.min
        max = dim_conf.max

        if dim_conf.integers_only:
            max = dim_conf.max + 0.99999

        # Update row with translated value
        row[i] = min + (max - min) * row[i]

        # Floor integer-only values
        if dim_conf.integers_only:
            row[i] = int(math.floor(row[i]))


columns = []
for dim_conf in design_config:
    columns.append(dim_conf.name)

df = pd.DataFrame(design, columns=columns)

print(df.tail(10))

df.to_excel('doe.xlsx', sheet_name='hypercube')
