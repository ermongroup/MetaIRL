from inverse_rl.algos.meta_irl_npo import MetaIRLNPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class MetaIRLTRPO(MetaIRLNPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(MetaIRLTRPO, self).__init__(optimizer=optimizer, **kwargs)
