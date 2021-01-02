r"""This module contains some baseline recommender systems.
"""
import os
from datetime import datetime

from rectorch.models import RecSysModel

__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ['Experiment']


class Experiment:
    """ Experiment Class TODO
    Parameters
    ----------
    models: array of :obj:`<rectorch.models.RecSysModel>`, required
        A collection of recommender models to evaluate, e.g., [C2PF, HPF, PMF].
    metrics: array of :obj:{`<cornac.metrics.RatingMetric>`, `<cornac.metrics.RankingMetric>`}, required
        A collection of metrics to use to evaluate the recommender models, \
        e.g., [NDCG, MRR, Recall].
    user_based: bool, optional, default: True
        This parameter is only useful if you are considering rating metrics. When True, first the average performance \
        for every user is computed, then the obtained values are averaged to return the final result.
        If `False`, results will be averaged over the number of ratings.
    show_validation: bool, optional, default: True 
        Whether to show the results on validation set (if exists).
        
    save_dir: str, optional, default: None
        Path to a directory for storing trained models and logs. If None, 
        models will NOT be stored and logs will be saved in the current working directory.
    Attributes
    ----------
    result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment 
        on the test set, initially it is set to None.
    
    val_result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the validation set (if exists), initially it is set to None.
    """

    def __init__(
        self,
        models,
        samplers,
        metrics,
        verbose=False,
        save_dir=None,
    ):
        self.models = self._validate_models(models)
        self.metrics = metrics
        self.verbose = verbose
        self.samplers = samplers
        self.result = None
        self.val_result = None

    @staticmethod
    def _validate_models(input_models):
        if not hasattr(input_models, "__len__"):
            raise ValueError(
                "models have to be an array but {}".format(type(input_models))
            )

        valid_models = []
        for model in input_models:
            if isinstance(model, RecSysModel):
                valid_models.append(model)
        return valid_models

    def _create_result(self):
        # TODO
        self.result = []
        pass
        # from ..eval_methods.cross_validation import CrossValidation
# 
        # if isinstance(self.eval_method, CrossValidation):
        #     self.result = CVExperimentResult()
        # else:
        #     self.result = ExperimentResult()
        #     if self.show_validation and self.eval_method.val_set is not None:
        #         self.val_result = ExperimentResult()

    def run(self):
        """Run the experiment"""
        self._create_result()
        
        for model, sampler in zip(self.models, self.samplers):
            # TODO: edit evaluate
            from rectorch.evaluation import evaluate
            from rectorch.utils import collect_results

            sampler.test()
            results = evaluate(model, sampler, self.metrics)

            self.result.append(results)
            print(collect_results(results))

        # output = ""
        # if self.val_result is not None:
        #     output += "\nVALIDATION:\n...\n{}".format(self.val_result)
        # output += "\nTEST:\n...\n{}".format(self.result)
# 
        # print(output)

        return self.result