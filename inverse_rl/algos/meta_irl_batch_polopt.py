import time

from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np
from collections import deque
from scipy.stats import pearsonr, spearmanr
from inverse_rl.utils.hyperparametrized import Hyperparametrized


class MetaIRLBatchPolopt(RLAlgorithm, metaclass=Hyperparametrized):
    """
    Base class for batch sampling-based meta policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            meta_batch_size=16,
            batch_size=10,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=True,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            init_pol_params = None,
            init_context_encoder_params=None,
            irl_model=None,
            irl_model_wt=1.0,
            discrim_train_itrs=10,
            zero_environment_reward=False,
            init_irl_params=None,
            train_irl=True,
            train_policy=True,
            train_context_only=False,
            pretrain_model=None,
            randomize_policy=False,
            key='',
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.meta_batch_size = meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.init_pol_params = init_pol_params
        self.init_irl_params = init_irl_params
        self.init_context_encoder_params = init_context_encoder_params
        self.irl_model = irl_model
        self.irl_model_wt = irl_model_wt
        self.no_reward = zero_environment_reward
        self.discrim_train_itrs = discrim_train_itrs
        self.train_irl = train_irl
        self.train_policy = train_policy
        self.pretrain_model = pretrain_model
        self.randomize_policy = randomize_policy
        self.train_context_only = train_context_only
        self.__irl_params = None
        self.pol_ret = []

        if self.irl_model_wt > 0:
            assert self.irl_model is not None, "Need to specify a IRL model"

        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                print('using vec sampler')
                sampler_cls = VectorizedSampler
            else:
                print('using batch sampler')
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        sampler_args['n_envs'] = self.meta_batch_size
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, policy_contexts=None):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)
        paths = self.sampler.obtain_samples(itr, reset_args, policy_contexts, return_dict=True)
        assert type(paths) == dict
        return paths

    def process_samples(self, itr, paths, log=False):
        return self.sampler.process_samples(itr, paths, log=log)

    def log_avg_returns(self, paths):
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        avg_return = np.mean(undiscounted_returns)
        return avg_return

    def get_irl_params(self):
        return self.__irl_params

    def compute_irl(self, paths, expert_traj_batch=None, itr=0):
        original_ret = []
        discrim_ret = []

        if self.no_reward:
            tot_rew = 0
            for key in paths.keys():
                for path in paths[key]:
                    original_ret.extend(path['rewards'])
                    tot_rew += np.sum(path['rewards'])
                    path['rewards'] *= 0
            logger.record_tabular('OriginalTaskAverageReturn', tot_rew/float(len(paths)*len(paths[key])))
            self.pol_ret.append(tot_rew/float(len(paths)*len(paths[key])))
            logger.record_tabular('OriginalTaskRewardMean', tot_rew/float(len(paths)*len(paths[key])*len(paths[key][0]['rewards'])))

        if self.irl_model_wt <=0:
            return paths

        if self.train_irl:
            max_itrs = self.discrim_train_itrs
            lr=1e-3
            self.irl_model.fit(paths, expert_traj_batch=expert_traj_batch, policy=self.policy, itr=itr, max_itrs=max_itrs, lr=lr,
                                           batch_size=int(self.batch_size/self.max_path_length/self.meta_batch_size), logger=logger,
                                           train_context_only=self.train_context_only)

            self.__irl_params = self.irl_model.get_params()

        probs = self.irl_model.eval(paths, expert_traj_batch=expert_traj_batch, gamma=self.discount, itr=itr)
        for key in probs.keys():
            for i, path in enumerate(probs[key]):
                if self.warm_up:
                    discrim_ret.extend(path)
                else:
                    if i < int(self.batch_size/self.max_path_length/self.meta_batch_size):
                        discrim_ret.extend(path)
        assert type(probs) is dict

        logger.record_tabular('IRLRewardMean', np.mean([probs[key] for key in probs.keys()]))
        logger.record_tabular('IRLRewardMax', np.max([probs[key] for key in probs.keys()]))
        logger.record_tabular('IRLRewardMin', np.min([probs[key] for key in probs.keys()]))
        try:
            logger.record_tabular('spearman', float(spearmanr(original_ret, discrim_ret)[0]))
            logger.record_tabular('pearsonr', float(pearsonr(original_ret, discrim_ret)[0]))
        except:
            import pdb; pdb.set_trace()


        if self.irl_model.score_trajectories:
            # TODO: should I add to reward here or after advantage computation?
            for key in paths.keys():
                for i, path in enumerate(paths[key]):
                    path['rewards'][-1] += self.irl_model_wt * probs[key][i]
        else:
            for key in paths.keys():
                for i, path in enumerate(paths[key]):
                    path['rewards'] += self.irl_model_wt * probs[key][i]
        return paths

    def train(self):
        # TODO: make this an util
        flatten_list = lambda l: [item for sublist in l for item in sublist]
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        if self.init_context_encoder_params is not None:
            self.irl_model.context_encoder.set_param_values(self.init_context_encoder_params)
        if self.init_pol_params is not None:
            self.policy.set_param_values(self.init_pol_params)
        if self.init_irl_params is not None:
            self.irl_model.set_params(self.init_irl_params)
        if self.pretrain_model is not None:
            if self.randomize_policy:
                policy_params = self.policy.get_param_values()
            self.pretrain_model.run(sess)
            if self.randomize_policy:
                self.policy.set_param_values(policy_params)
        self.start_worker()
        start_time = time.time()

        returns = []
        expert_obs, expert_acts, expert_contexts = self.irl_model.extract_paths(self.irl_model.expert_trajs,
                                                    keys=('observations', 'actions', 'contexts'), T=self.max_path_length)
        expert_trajs = np.concatenate((expert_obs, expert_acts), axis=-1) # num_experts x T x (state_dim + act_dim)
        self.warm_up = True
        warm_up_step = int(len(self.irl_model.expert_trajs) / self.meta_batch_size)
        warm_up_idx = 0
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Sampling set of expert trajs for this meta-batch...")
                expert_traj_batch, m_batch = self.irl_model.sample_batch(expert_trajs, expert_contexts, batch_size=self.meta_batch_size, 
                    warm_up=self.warm_up, warm_up_idx = warm_up_idx)
                warm_up_idx = (self.meta_batch_size + warm_up_idx) % len(self.irl_model.expert_trajs)
                if itr >= warm_up_step:
                    self.warm_up = False
                m_hat_batch, _ = self.irl_model.context_encoder.get_actions(expert_traj_batch.reshape(-1, self.max_path_length*(self.irl_model.dO+self.irl_model.dU)))

                logger.log("Sampling set of tasks/goals for this meta-batch based on true m...")

                env = self.env
                learner_env_goals = list(m_batch[:, 0, :])

                logger.log("Obtaining samples...")
                # Use reset args to reset the environment and use policy contexts to serve as part of the observation
                paths = self.obtain_samples(itr, reset_args=learner_env_goals, policy_contexts=list(m_hat_batch))

                logger.log("Processing samples...")
                paths = self.compute_irl(paths, expert_traj_batch=expert_traj_batch, itr=itr)
                # returns.append(self.log_avg_returns(paths))

                samples_data = {}
                for key in paths.keys():  # the keys are the tasks
                    # don't log because this will spam the consol with every task.
                    samples_data[key] = self.process_samples(itr, paths[key], log=False)
                # for logging purposes only
                self.process_samples(itr, flatten_list(paths.values()), log=True)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(flatten_list(paths.values()))

                logger.log("Optimizing policy...")
                if self.train_policy:
                    self.optimize_policy(itr, samples_data, expert_traj_batch)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = {key:samples_data[key]["paths"] for key in samples_data.keys()}
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()
        return 

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
