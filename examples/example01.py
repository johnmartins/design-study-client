import pandas as pd
import sklearn.preprocessing as sklearn_preproc

from prediction.metamodel import build_gaussian_process, evaluate_metamodel, build_response_surface

df = pd.read_excel('./examples/example_assets/doe.xlsx')

input_cols = ["VANE_TOTAL_COUNT", "VANE_LEAN_RAD", "VANE_LE_RADIUS", "VANE_CORDA_LENGTH", "T_VANE_REG",
              "T_VANE_MNT", "T_HUB_REG", "T_HUB_MNT", "T_OUTER_REG", "T_OUTER_MNT"]

output_col_deform = 'Max Deformation [m]'
output_col_volume = 'Volume'
output_col_th_stress = 'Thermal Stress [MPa]'
output_cols = [output_col_deform, output_col_volume, output_col_th_stress]

df = df.dropna(subset=[output_col_deform, output_col_volume, output_col_th_stress])
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

metamodel_volume = build_gaussian_process(df_train, input_cols, output_col_volume)
# metamodel_volume = build_response_surface(df_train, input_columns=input_cols, output_column=output_col_volume)
mm_eval = evaluate_metamodel(df_test, metamodel_volume, input_columns=input_cols, output_column=output_col_volume)
df_eval = mm_eval[0]
df_eval[output_cols] = scaler_y.inverse_transform(df_eval[output_cols])
df_eval[input_cols] = scaler_x.inverse_transform(df_eval[input_cols])
df_eval[["$PREDICTION$"]] = df_original[output_col_volume].min() + (df_eval[["$PREDICTION$"]] * (df_original[output_col_volume].max() - df_original[output_col_volume].min()))
df_eval[["ERROR"]] = (df_eval[["ERROR"]] * (df_original[output_col_volume].max() - df_original[output_col_volume].min()))

print(df_eval.to_string())

prediction = metamodel_volume.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

