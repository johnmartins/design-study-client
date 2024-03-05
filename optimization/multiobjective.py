import numpy as np
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination, MixedVariableGA
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from setup.models import DimensionConfiguration


class Problem(ElementwiseProblem):

    def __init__(self, metamodels: list, dimension_config_list: list[DimensionConfiguration], **kwargs):
        n_obj = len(metamodels)

        self.metamodels = metamodels
        self.var_list = []
        self.dimension_config_list = dimension_config_list
        variables = Problem.convert_variables_to_pymoo_format(dimension_config_list)

        for col in variables.keys():
            self.var_list.append(col)

        super().__init__(vars=variables, n_obj=n_obj, n_ieq_constr=0, **kwargs)

    @staticmethod
    def convert_variables_to_pymoo_format(dimension_config_list: list[DimensionConfiguration]):
        variables = dict()

        for dc in dimension_config_list:
            if dc.integers_only:
                variables[dc.name] = Integer(bounds=[dc.optimization_min, dc.optimization_max])
            else:
                variables[dc.name] = Real(bounds=[dc.optimization_min, dc.optimization_max])

        return variables

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array([x[k] for k in self.var_list])
        f_outputs = []
        for mm in self.metamodels:
            f_outputs.append(mm.predict([x]))

        out["F"] = f_outputs


def optimize(problem: ElementwiseProblem, pop_size: int, n_gen: int = 100, seed: int = 1, verbose: bool = True):
    algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival())
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=verbose)
    return res.X, res.F
