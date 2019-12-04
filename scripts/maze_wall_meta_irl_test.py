import tensorflow as tf

from inverse_rl.algos.meta_irl_trpo import MetaIRLTRPO
from inverse_rl.models.info_airl_state_test import *
from inverse_rl.models.tf_util import get_session_config
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.base import EnvSpec
from sandbox.rocky.tf.spaces.box import Box
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.models.tf_util import load_prior_params
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial, run_sweep_parallel_custom
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts, load_latest_experts_multiple_runs
import pickle


DATA_DIR = 'data_fusion_discrete/maze_wall_meta_irl_imitcoeff-0.01_infocoeff-0.1_mbs-50_bs-16_itr-20_preepoch-1000_entropy-1.0_RandomPol_Rew-2-32/2019_05_14_02_33_17_0'


def main(exp_name=None, latent_dim=3, params_folder=None):
    max_path_length = 100
    batch_size = 32
    meta_batch_size = 50
    entropy_weight = 0.1
    left = 'right'
    if_filtered = True

    # tf.reset_default_graph()
    if left == 'left':
        env = TfEnv(CustomGymEnv('PointMazeLeft-v0', record_video=False, record_log=False))
    else:
        env = TfEnv(CustomGymEnv('PointMazeRight-v0', record_video=False, record_log=False))

    # load ~2 iterations worth of data from each forward RL experiment as demos
    experts = load_latest_experts_multiple_runs('/atlas/u/lantaoyu/projects/InfoAIRL/data/maze_left_data_collect', n=4, latent_dim=latent_dim)
    if if_filtered:
        experts_filtered = []
        good_range = [0.1, 0.4]#[0.3, 0.5]
        for expert in experts:
            if expert['contexts'][0,0] >= good_range[0] and expert['contexts'][0,0] <= good_range[1]:
                experts_filtered.append(expert)
        assert len(experts_filtered) >= meta_batch_size
        experts_filtered = experts_filtered[:-(len(experts_filtered)%meta_batch_size)]
        experts = experts_filtered 

    irl_itr_list = [2800]

    results = []
    for irl_itr in irl_itr_list:
        params_file = os.path.join(DATA_DIR, 'itr_%d.pkl' % irl_itr)
        prior_params = load_prior_params(params_file)
        init_context_encoder_params = load_prior_params(params_file, 'context_params')
        
        policy_prior_params = None

        # contexual policy pi(a|s,m)
        policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

        # approximate posterior q(m|tau)
        context_encoder_spec = EnvSpec(
            observation_space=Box(np.tile(np.concatenate((env.observation_space.low[:-latent_dim], env.action_space.low)), max_path_length),
                                    np.tile(np.concatenate((env.observation_space.high[:-latent_dim], env.action_space.high)), max_path_length)),
            action_space=Box(np.zeros(latent_dim), np.ones(latent_dim)),
        )
        context_encoder = GaussianMLPPolicy(name='context_encoder', env_spec=context_encoder_spec, hidden_sizes=(128, 128))

        irl_model = InfoAIRL(env=env,
                            expert_trajs=experts,
                            context_encoder=context_encoder,
                            state_only=True,
                            max_path_length=max_path_length,
                            meta_batch_size=meta_batch_size,
                            latent_dim=latent_dim)

        algo = MetaIRLTRPO(
            init_irl_params=prior_params,
            init_pol_params=policy_prior_params,#policy_prior_params,
            init_context_encoder_params=init_context_encoder_params,
            env=env,
            policy=policy,
            irl_model=irl_model,
            n_itr=150,
            meta_batch_size=meta_batch_size,
            batch_size=batch_size,
            max_path_length=max_path_length,
            discount=0.99,
            store_paths=True,
            train_irl=True, # True
            train_context_only=True,
            train_policy=True,
            irl_model_wt=1.0,
            entropy_weight=entropy_weight,
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            log_params_folder=params_folder,
            log_experiment_name=exp_name,
        )
        with rllab_logdir(algo=algo, dirname='data_finetune/maze_finetune_discrete-entropy-%s-irl_itr-%s-%s-%s-generalize/%s' % (entropy_weight, irl_itr, left, 'filter' if if_filtered else '', exp_name)):
            with tf.Session():
                algo.train()
        results.append((irl_itr, np.max(algo.pol_ret))) 
        tf.reset_default_graph()
    print(results)


if __name__ == "__main__":
    import os
    # params_folders = DATA_DIR #os.listdir(DATA_DIR)
    params_dict = {
        # 'params_folder': [params_folders[1]],
        'latent_dim': [3],
    }
    # results = run_sweep_parallel(main, params_dict, repeat=1)
    run_sweep_serial(main, params_dict, repeat=1)
