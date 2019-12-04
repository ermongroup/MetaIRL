import tensorflow as tf

from inverse_rl.algos.trpo import TRPO
from inverse_rl.models.tf_util import get_session_config
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial


def main(exp_name, ent_wt=1.0, discrete=True):
    tf.reset_default_graph()
    if discrete:
        env = TfEnv(CustomGymEnv('PointMazeLeft-v0', record_video=False, record_log=False))
    else:
        env = TfEnv(CustomGymEnv('PointMazeLeftCont-v0', record_video=False, record_log=False))

    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    with tf.Session(config=get_session_config()) as sess:
        algo = TRPO(
            env=env,
            sess=sess,
            policy=policy,
            n_itr=2000,
            batch_size=20000,
            max_path_length=500,
            discount=0.99,
            store_paths=True,
            entropy_weight=ent_wt,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            exp_name=exp_name,
        )
        if discrete:
            output = 'data/maze_left_data_collect_discrete-15/%s' % exp_name
        else:
            output = 'data/maze_left_data_collect/%s' % exp_name
        with rllab_logdir(algo=algo, dirname=output):
            algo.train()


if __name__ == "__main__":
    params_dict = {
        'ent_wt': [0.1],
        'discrete': True # Setting discrete to 'True' to get training data, 'False' to get test data (test unseen positions)
    }
    run_sweep_parallel(main, params_dict, repeat=4)
