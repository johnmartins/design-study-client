import pandas as pd
import sklearn.preprocessing as sklearn_preproc
import numpy as np
import matplotlib.pyplot as plt

from setup.models import DimensionConfiguration
from prediction.metamodel import build_gaussian_process, evaluate_metamodel, build_response_surface
from optimization.multiobjective import Problem, optimize

df = pd.read_excel('./examples/example_assets/doe.xlsx')

design_config = [
    DimensionConfiguration(name='VANE_TOTAL_COUNT', min=8, max=18, integers_only=True,
                           optimization_min=0, optimization_max=1),
    DimensionConfiguration(name='VANE_LEAN_RAD', min=0, max=(20/180)*np.pi,
                           optimization_min=0, optimization_max=1),
    DimensionConfiguration(name='VANE_LE_RADIUS', min=12, max=19,
                           optimization_min=0, optimization_max=1),
    DimensionConfiguration(name='VANE_CORDA_LENGTH', min=1.4, max=3,
                           optimization_min=0, optimization_max=1),
    DimensionConfiguration(name='T_VANE_REG', min=1.4, max=3,
                           optimization_min=0, optimization_max=1),
    DimensionConfiguration(name='T_VANE_MNT', min=1.4, max=3,
                           optimization_min=0, optimization_max=1),
    DimensionConfiguration(name='T_HUB_REG', min=1.4, max=3,
                           optimization_min=0, optimization_max=1),
    DimensionConfiguration(name='T_HUB_MNT', min=1.4, max=3,
                           optimization_min=0, optimization_max=1),
    DimensionConfiguration(name='T_OUTER_REG', min=1.4, max=3,
                           optimization_min=0, optimization_max=1),
    DimensionConfiguration(name='T_OUTER_MNT', min=1.4, max=3,
                           optimization_min=0, optimization_max=1),
]

input_cols = []
for dc in design_config:
    input_cols.append(dc.name)

output_col_deform = 'Max Deformation [m]'
output_col_volume = 'Volume'
output_col_th_stress = 'Thermal Stress [MPa]'
output_cols = [output_col_deform, output_col_volume, output_col_th_stress]

df = df.dropna()
df.drop(df[(df['Volume'] < 0.1)].index, inplace=True)

df_original = df.copy()

# Normalize (scale) inputs and outputs to a range of [0, 1]
scaler_x = sklearn_preproc.MinMaxScaler(feature_range=(0, 1))
df[input_cols] = scaler_x.fit_transform(df[input_cols])
scaler_y = sklearn_preproc.MinMaxScaler(feature_range=(0, 1))
df[output_cols] = scaler_y.fit_transform(df[output_cols])

# Extract training and test set
df_train = df.sample(frac=0.97)
df_test = df.loc[~df.index.isin(df_train.index)]

# Build metamodels
metamodels = []
for output_col in output_cols:
    metamodel = build_gaussian_process(df_train, input_cols, output_col)
    evaluate_metamodel(df_test, metamodel, input_columns=input_cols, output_column=output_col)
    metamodels.append(metamodel)

"""
df_eval = mm_eval[0]
df_eval[output_cols] = scaler_y.inverse_transform(df_eval[output_cols])
df_eval[input_cols] = scaler_x.inverse_transform(df_eval[input_cols])
df_eval[["$PREDICTION$"]] = df_original[output_col_volume].min() + (df_eval[["$PREDICTION$"]] * (df_original[output_col_volume].max() - df_original[output_col_volume].min()))
df_eval[["ERROR"]] = (df_eval[["ERROR"]] * (df_original[output_col_volume].max() - df_original[output_col_volume].min()))
"""

# Optimize
optimization_problem = Problem(metamodels, design_config)
X_star, F_star = optimize(optimization_problem, 1000, n_gen=50)

# Reformat the data so that Pandas can consume it
X_star_array = []
for dp in X_star:
    ar = np.array([dp[col] for col in input_cols])
    X_star_array.append(ar)

# Build dataframe with optimized inputs and outputs ("star" values)
df_star = pd.DataFrame()
df_star[input_cols] = scaler_x.inverse_transform(X_star_array)
df_star[output_cols] = scaler_y.inverse_transform(F_star)

print(df_star.to_string())

print(df_original.to_string())

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(df_star[output_col_deform], df_star[output_col_volume], df_star[output_col_th_stress], marker='o')
# plt.figure(figsize=(7, 5))
# plt.scatter(df_star[output_col_deform], df_star[output_col_volume], s=30, facecolors='none', edgecolors='blue')
plt.gray()
plt.show()

df_star.to_excel('Ruttifrutti.xlsx')
