import numbers
from itertools import product

import numpy as np
from sympy import Matrix, Symbol
from sympy.core import Float


def set_difference(lst1, lst2):
    """returns the elements and indicies of elements in lst1 that are not in lst2"""
    elements = []
    indicies = []
    for indx, item in enumerate(lst1):
        if item not in lst2:
            elements.append(item)
            indicies.append(indx)
    return elements, indicies


def to_symbolic(lst):
    all_numeric = True
    result = []
    for item in lst:
        try:
            result.append(Symbol(item))
            all_numeric = False
        except TypeError:
            result.append(item)
    return result, all_numeric


class DiscreteFactor(object):
    def __init__(self, variables, card, data):
        """
        variables: name of variables in factor
        card: cardinality of variables (in same order as variables list)
        data: data for the factor
        """
        if len(variables) != len(card):
            raise ValueError("variables and card must have the same length")
        expected_len_data = np.prod(card)
        if len(data) != expected_len_data:
            raise ValueError(
                "length of data array inconsistent with cardinality, should be {0}".format(
                    expected_len_data
                )
            )
        self.variables = variables
        self.card = card
        self.data, _ = to_symbolic(data)
        assignments = product(*[range(c) for c in card])
        var_assignments = [
            sorted(zip(variables, a), key=lambda x: x[0]) for a in assignments
        ]
        keys = [
            "".join([s[0] + str(s[1]) for s in assignment])
            for assignment in var_assignments
        ]
        self.key2index = dict(zip(keys, range(len(data))))

    def __str__(self):
        return self.data.__str__()

    def probability_of(self, assignment):
        """
        assignment: list of tuples (variable, value)
        return the probability of this assignment
        """
        key = "".join(
            [s[0] + str(s[1]) for s in sorted(assignment, key=lambda x: x[0])]
        )
        indx = self.key2index[key]
        return self.data[indx]

    def multiply(self, other):
        # TODO check cardinality of matching vars is the same
        new_vars, indices = set_difference(other.variables, self.variables)
        new_card = [other.card[i] for i in indices]
        result_vars = self.variables + new_vars
        result_card = self.card + new_card
        num_vars = len(self.variables)
        other_var_indicies = [
            i for i, var in enumerate(result_vars) if var in other.variables
        ]
        assignments = product(*[range(c) for c in result_card])

        result_data = []
        for a in assignments:

            assignment1 = zip(self.variables, a[0:num_vars])
            assignment2 = zip(other.variables, [a[i] for i in other_var_indicies])
            p1 = self.probability_of(assignment1)
            p2 = other.probability_of(assignment2)
            result_data.append(p1 * p2)
        return DiscreteFactor(result_vars, result_card, result_data)

    def condition(self, variables, values):
        return 1


class DiscreteBN(object):
    def __init__(self):
        self._variables = []
        self._parents = {}
        self._card = {}
        self._conditionals = []
        self.joint = None

    def add_var(self, variable, cardinality, parents=None, prob_table=None):
        """
        variable: name of variable
        cardinality: the number of values this variable can take
        parents: A list of the names of the parents of this variable
        prob_table: A table for the conditional probability of this variable given its parents (in order listed)

        """
        if parents is None:
            parents = []
        for p in parents:
            if p not in self._variables:
                raise ValueError(
                    "Parent {p} is not a variable in the network".format(p=p)
                )

        card = [cardinality] + [self._card[parent] for parent in parents]
        variables = [variable] + parents
        prob_table_length = np.prod(card)  # product of cardinality of parents

        if prob_table is None:
            assignments = product(*[range(c) for c in card])
            prob_table = [self._p_symbol(variable, a) for a in assignments]

        if len(prob_table) != prob_table_length:
            raise ValueError(
                "The probability table has length {0}, but should have length {1}".format(
                    len(prob_table), prob_table_length
                )
            )

        conditional = DiscreteFactor(variables, card, prob_table)
        self._variables.append(variable)
        self._parents[variable] = parents
        self._card[variable] = cardinality
        self._conditionals.append(conditional)
        if self.joint is None:
            self.joint = conditional
        else:
            self.joint = self.joint.multiply(conditional)

    def probability(self, left, right=None):
        """returns the (conditional probablity) of the variables on the left given the variables on the right"""
        return 1

    def _p_symbol(self, variable, assignment):
        return Symbol("P" + variable + "_" + "".join([str(a) for a in assignment]))


class LinearGaussianBN(object):
    """
    A Bayesian network in which all variables are linear functions of their parents plus additive gaussian noise.
    The joint distribution over the variables in such a network is a multivariate gaussian.
    """

    def __init__(self):
        self._variables = []
        self._mean = None
        self._cov = None
        self._parents = {}
        self._weights = {}
        self._variance = {}
        self._weights_p = {}
        self._variance_p = {}

    def copy(self):
        model = LinearGaussianBN()
        model._variables = list(self._variables)
        model._mean = (
            None if self._mean is None else self._mean[:, :]
        )  # this is a Matrix, slice = copy
        model._cov = None if self._cov is None else self._cov[:, :]
        model._parents = self._parents.copy()
        model._weights = self._weights.copy()
        model._variance = self._variance.copy()
        model._weights_p = self._weights_p.copy()
        model._variance_p = self._variance_p.copy()
        return model

    def __str__(self):
        result = ""
        for variable in self.variables:
            parent_list = self.parents(variable)
            weights = self.get_weights(variable)
            equation = str(weights[0])
            for i, p in enumerate(parent_list):
                w = weights[i + 1]
                if isinstance(w, Float):
                    w = float(w)
                try:
                    equation += " + {0:g}*{1}".format(w, p)
                except ValueError:
                    equation += " + {0}*{1}".format(w, p)

            if equation.startswith("0 +"):
                equation = equation[4:]

            v = self.get_variance(variable)
            result += "{0} ~ N({1} ; {2})\n".format(variable, equation, v)
        return result

    def get_variance(self, variable):
        return self._variance[variable]

    def get_weights(self, variable):
        return self._weights[variable]

    def _variance_symbol(self, variable):
        """returns a symbol for the marginal variance of variable string specified."""
        return Symbol("e_{0}".format(variable))

    def _weight_symbol(self, variable, parent):
        if parent is not None:
            return Symbol("w_" + variable + parent)
        return Symbol("w_" + variable + "0")

    def add_var(self, variable, parents=None, weights=None, variance=None):
        """
        Add a variable to the network.
        - variable: a string representing the variable
        - parents: a list of the names of the parents of this variable (must already be in the network)
        - weights (optional): the weights of this variable to its parents. The first entry should be the offset
        - variance: the variance of this variable after conditioning on its parents values.
        """
        if parents is None:
            parents = []
        for p in parents:
            if p not in self._variables:
                raise ValueError(
                    "Parent {p} is not a variable in the network".format(p=p)
                )
        if variable in self._variables:
            raise ValueError(
                "Duplicate variable name, variable {v} already exists in this network".format(
                    v=variable
                )
            )

        if variance is None:
            variance = self._variance_symbol(variable)

        if weights is None:
            beta = [
                self._weight_symbol(variable, v) if v in parents else 0
                for v in self._variables
            ]
            mu = self._weight_symbol(variable, None)
            weights = [mu] + [self._weight_symbol(variable, v) for v in parents]

        else:
            if len(weights) != len(parents) + 1:
                raise ValueError(
                    """The vector of weights has length {0} but should be of length {1},
                                     (offset, weight_parent_1, weight_parent_2, ...,weight_parent_n)""".format(
                        len(weights), len(parents) + 1
                    )
                )

            weights = [
                w if isinstance(w, numbers.Number) else Symbol(w) for w in weights
            ]
            symbol_dict = dict(zip(parents, weights[1:]))
            beta = [symbol_dict[v] if v in parents else 0 for v in self.variables]
            mu = weights[0]

        v = variance
        if len(beta) > 0:
            beta = Matrix(beta)

            mu += (beta.T * self._mean)[0, 0]
            cv = self._cov * beta  # covariance of this variable with previous variables
            v += (beta.T * cv)[0, 0]  # variance of this variable (unconditional)

            new_vals = Matrix([cv, [v]])
            rows, cols = self._cov.shape
            self._cov = self._cov.row_insert(rows, cv.T)
            self._cov = self._cov.col_insert(cols, new_vals)
            self._mean = Matrix([self._mean, [mu]])

        else:  # first time round - everything is None
            self._cov = Matrix([v])
            self._mean = Matrix([mu])

        self._weights[variable] = Matrix(
            weights
        )  # order is with respect to parents as specified (not covariance matrix index)
        self._variance[variable] = variance
        self._parents[variable] = parents
        self._variables.append(variable)

        try:  # if the parameters are numeric, set current parameterisation.
            self.set_var_params(variable, weights, variance)
        except ValueError:
            pass

    @property
    def mu(self):
        return self._mean

    @property
    def cov(self):
        """the covariance matrix of the multivariate gaussian corresponding to the network"""
        return self._cov

    @property
    def variables(self):
        """The list of variables in the order coresponding to their position in the mean vector/covariance matrix"""
        return self._variables

    @property
    def information_matrix(self):
        """The inverse of the covariance matrix"""
        return self._cov.inv()

    def index(self, variables):
        """returns the indexes of the specified variables in the mean vector/covariance matrix"""
        return [self._variables.index(v) for v in variables]

    def marginal(self, variables):
        """returns a new network with all but the specified variables marginalized out"""
        indx = self.index(variables)
        mu = self.mu.extract(indx, [-1])
        cov = self.cov.extract(indx, indx)
        marginalized = self.copy()
        marginalized._variables = variables
        marginalized._mean = mu
        marginalized._cov = cov
        return marginalized

    def regression_matrix(self, condition_on):
        b_indx = self.index(condition_on)
        a_vars = [v for v in self._variables if v not in condition_on]
        a_indx = self.index(a_vars)

        cov_ab = self.cov.extract(a_indx, b_indx)
        cov_bb_inv = self.cov.extract(b_indx, b_indx).inv()
        return cov_ab * cov_bb_inv

    def observe(self, variables, values):
        """network where we condition on variables equaling some values"""
        sym_values = [Symbol(val) if isinstance(val, str) else val for val in values]
        values = Matrix(sym_values)
        # partition variables into A (not conditioned on), and B (conditioned on)
        a_vars = [v for v in self._variables if v not in variables]
        a_indx = self.index(a_vars)
        b_indx = self.index(variables)
        cov_aa = self.cov.extract(a_indx, a_indx)
        cov_ab = self.cov.extract(a_indx, b_indx)
        cov_bb = self.cov.extract(b_indx, b_indx)
        cov_bb_inv = cov_bb.inv()
        mu_a = self.mu.extract(a_indx, [-1])
        mu_b = self.mu.extract(b_indx, [-1])

        reg_matrix = cov_ab * cov_bb_inv

        mu = mu_a + reg_matrix * (values - mu_b)  # .applyfunc(lambda x:x.simplify())
        cov = cov_aa - reg_matrix * cov_ab.T  # .applyfunc(lambda x:x.simplify())

        observed = self.copy()
        observed._variables = a_vars
        observed._mean = mu
        observed._cov = cov
        return observed

    def do(self, variables, values):
        """network after intervening to set specified variables to values"""
        intervened = LinearGaussianBN()
        sym_values = [Symbol(val) if isinstance(val, str) else val for val in values]
        var_to_value = dict(zip(variables, sym_values))
        for variable in self.variables:
            if variable not in variables:
                new_parents = []
                new_weights = [self._weights[variable][0]]
                for indx, p in enumerate(self.parents(variable)):
                    w = self._weights[variable][indx + 1]
                    if p not in variables:
                        new_parents.append(p)
                        new_weights.append(w)
                    else:
                        new_weights[0] += w * var_to_value[p]

                intervened.add_var(variable, new_parents, new_weights)

        return intervened

    def parents(self, variable):
        return self._parents[variable]

    def set_var_params(self, variable, weights, variance):
        """
        Set numerical values for the weights and variance of the specified variable.
        first weight is assumed to be constant.
        Other weights are with respect to parents in the same order as they were originally specified
        (or as returned by parents(variable))
        """
        if variable not in self._weights:
            raise ValueError("Variable {0} not in network".format(variable))

        if not all(isinstance(w, numbers.Number) for w in weights):
            raise ValueError("Weights are not all numeric, {0}".format(weights))

        if not isinstance(variance, numbers.Number):
            raise ValueError("Variance is not numeric")

        expected_num_weights = len(self._weights[variable])
        if len(weights) != expected_num_weights:
            raise ValueError(
                "Weights should contain {0} values (number of parents + 1) but contained {1} values"
            ).format(expected_num_weights, len(weights))
        self._weights_p[variable] = weights
        self._variance_p[variable] = variance

    def set_params(self, variable_param_dict):
        """
        Set numerical values for all variables based on dictionary.
        The dictionary keys should be variable names and the values a tuple containing (weights, variance), eg
        {"Z":([0],1), "X":([0,.5],.2)}
        """
        for variable, (weights, variance) in variable_param_dict.items():
            self.set_var_params(variable, weights, variance)
        return self

    def parameterized_mean_cov(self):
        """Returns the mean and covariance after substituting parameters. All variables must have parameters set."""
        substitutions = []
        for variable, sym_weights in self._weights.items():
            numeric_weights = self._weights_p[variable]
            substitutions.extend(list(zip(sym_weights, numeric_weights)))
            substitutions.append((self._variance[variable], self._variance_p[variable]))
        cov = self._cov.subs(substitutions)
        mu = self._mean.subs(substitutions)
        return mu, cov

    def sample(self, n):
        """sample from the underlying multivariate gaussain. All variables must have parameters set."""
        unparameterized = [
            v for v in self._weights.keys() if v not in self._weights_p.keys()
        ]
        if len(unparameterized) > 0:
            raise ValueError(
                "The following variables must be numerically parameterized before sampling: {0}".format(
                    unparameterized
                )
            )
        mu, cov = self.parameterized_mean_cov()
        cov = np.asarray(cov).astype(np.float64)
        mu = np.asarray(mu).astype(np.float64)
        return np.random.multivariate_normal(mu.ravel(), cov, size=n)

    # def sample2(self,n):
    #    data_dict = {}
    #    for v in self.variables:
    #        parent_list = self.parents(variable)
    #        weights = self.weights_p[variable]
    #        data = np.random.normal(mu,variance,size=n)


# model = LinearGaussianBN()
# model.add_var("U",None,[0],1)
# model.add_var("Z",["U"],[0,2],1)
# model.add_var("X",["Z"],[0,.5],2)
# model.add_var("Y",["U","Z","X"],[0,3,-1,2],1)
#
# print model
# model.sample(10)
#
# model = LinearGaussianBN()
# model.add_var("U",[])
# model.add_var("Z",["U"])
# model.add_var("X",["Z"])
# model.add_var("Y",["U","Z","X"])
#
# print model
