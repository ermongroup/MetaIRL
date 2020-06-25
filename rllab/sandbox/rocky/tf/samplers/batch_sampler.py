from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
import tensorflow as tf


def worker_init_tf(G):
    G.sess = tf.Session()
    G.sess.__enter__()


def worker_init_tf_vars(G):
    G.sess.run(tf.global_variables_initializer())


class BatchSampler(BaseSampler):
    def __init__(self, algo, n_envs=1):
        super(BatchSampler, self).__init__(algo)
        self.n_envs = n_envs

    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.algo.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr, reset_args=None, policy_contexts=None, return_dict=False):
        init_policy_params = cur_policy_params = self.algo.policy.get_param_values()
        if hasattr(self.algo.env,"get_param_values"):
            try:
                cur_env_params = self.algo.env.get_param_values()
            except:
                cur_env_params = None
        else:
            cur_env_params = None
        import time
        start = time.time()
        if type(reset_args) != list and type(reset_args)!=np.ndarray:
            reset_args = [reset_args]*self.n_envs
        if type(policy_contexts) != list and type(policy_contexts)!=np.ndarray:
            policy_contexts = [policy_contexts]*self.n_envs
        cur_policy_params = [cur_policy_params]*self.n_envs
        # do tasks sequentially and parallelize within rollouts per task.
        paths = {}
        for i in range(self.n_envs):
            paths[i] = parallel_sampler.sample_paths(
                policy_params=cur_policy_params[i],
                env_params=cur_env_params,
                max_samples=self.algo.batch_size / self.n_envs,
                max_path_length=self.algo.max_path_length,
                scope=self.algo.scope,
                reset_arg=reset_args[i],
                policy_context=policy_contexts[i],
                show_prog_bar=False,
            )
        total_time = time.time() - start
        logger.record_tabular("TotalExecTime", total_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())

        self.algo.policy.set_param_values(init_policy_params)

        # currently don't support not whole paths (if desired, add code to truncate paths)
        assert self.algo.whole_paths

        return paths

    # def start_worker(self):
    #     if singleton_pool.n_parallel > 1:
    #         singleton_pool.run_each(worker_init_tf)
    #     parallel_sampler.populate_task(self.algo.env, self.algo.policy)
    #     if singleton_pool.n_parallel > 1:
    #         singleton_pool.run_each(worker_init_tf_vars)

    # def shutdown_worker(self):
    #     parallel_sampler.terminate_task(scope=self.algo.scope)

    # def obtain_samples(self, itr):
    #     cur_policy_params = self.algo.policy.get_param_values()
    #     cur_env_params = self.algo.env.get_param_values()
    #     paths = parallel_sampler.sample_paths(
    #         policy_params=cur_policy_params,
    #         env_params=cur_env_params,
    #         max_samples=self.algo.batch_size,
    #         max_path_length=self.algo.max_path_length,
    #         scope=self.algo.scope,
    #     )
    #     if self.algo.whole_paths:
    #         return paths
    #     else:
    #         paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
    #         return paths_truncated
