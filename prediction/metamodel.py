import numpy as np

from sklearn import linear_model
import sklearn.preprocessing as sklearn_preproc
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def build_metamodel(df_train, input_cols, output_column, meta_model_type='gp', response_surface_degree=2,
                    response_surface_fit_intercept=False, gp_rbf_length_scale=1):
    """

    :param df_train: Pandas dataframe containing the training data
    :param input_cols: Array of column names that represent the inputs of the dataset
    :param output_column: Name of the column which the metamodel will be trained to predict.
    :param meta_model_type: Can be "gp" (gaussian process with RBF kernel) or "rs" (response surface)
    :param response_surface_degree: Defaults to 2
    :param response_surface_fit_intercept: Defaults to False
    :param gp_rbf_length_scale: Defaults to 1.
    :return:
    """

    if meta_model_type == 'rs':
        return build_response_surface(df_train, input_cols, output_column, deg=response_surface_degree,
                                      fit_intercept=response_surface_fit_intercept)
    if meta_model_type == 'gp':
        return build_gaussian_process(df_train, input_cols, output_column, rbf_length_scale=gp_rbf_length_scale)
    else:
        raise ValueError(f'Unsupported meta model type "{meta_model_type}"')


def build_response_surface(df_train, input_columns, output_column, deg=2, fit_intercept=False):
    x = df_train[input_columns].values
    y = df_train[[output_column]].values
    metamodel = Pipeline([('poly', sklearn_preproc.PolynomialFeatures(degree=deg)),
                          ('linear', linear_model.LinearRegression(fit_intercept=fit_intercept))])
    metamodel.fit(x, y)
    return metamodel


def build_gaussian_process(df_train, input_columns, output_column, rbf_length_scale=1):
    x = df_train[input_columns].values
    y = df_train[[output_column]].values

    kernel = RBF(length_scale=rbf_length_scale, length_scale_bounds="fixed")
    metamodel = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    metamodel.fit(x, y)
    return metamodel


def evaluate_metamodel(df_test, metamodel, input_columns, output_column, error_calc='absolute',
                       prediction_column=f"$PREDICTION$"):
    """
    Evaluate meta model error using a test dataset
    :param df_test:
    :param prediction_column: The
    :param metamodel: The metamodel that is to be tested
    :param input_columns:
    :param output_column:
    :param error_calc: Can be "absolute" or "squared"
    :return: (min_error, max_error, mean_error, std_dev_error)
    """
    df_test[prediction_column] = metamodel.predict(df_test[input_columns])

    if error_calc == 'absolute':
        df_test['ERROR'] = np.sqrt((df_test[output_column] - df_test[prediction_column])**2)
    elif error_calc == 'squared':
        df_test['ERROR'] = (df_test[output_column] - df_test[prediction_column])**2
    else:
        raise ValueError(f'Unknown error calculation type {error_calc}')

    # Prediction analysis
    max_error = df_test['ERROR'].max()
    min_error = df_test['ERROR'].min()
    mean_error = df_test['ERROR'].mean()
    std_dev_error = df_test['ERROR'].std()

    print(f'Min error:\t\t\t{min_error}')
    print(f'Max error:\t\t\t{max_error}')
    print(f'Mean error:\t\t\t{mean_error}')
    print(f'Standard deviation (error):\t{std_dev_error}')

    return min_error, max_error, mean_error, std_dev_error
