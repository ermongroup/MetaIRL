import tensorflow as tf

from inverse_rl.algos.meta_irl_trpo import MetaIRLTRPO
from inverse_rl.models.info_airl_state_test import *
from inverse_rl.models.tf_util import get_session_config
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.base import EnvSpec
from sandbox.rocky.tf.spaces.box import Box
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from inverse_rl.models.architectures import relu_net, linear_net
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.models.tf_util import load_prior_params
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial, run_sweep_parallel_custom
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts, load_latest_experts_multiple_runs
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


DATA_DIR = 'data_fusion_discrete/maze_wall_meta_irl_imitcoeff-0.01_infocoeff-0.1_mbs-50_bs-16_itr-20_preepoch-1000_entropy-1.0_RandomPol_Rew-2-32/2019_05_14_02_33_17_0'


def main(exp_name=None, latent_dim=3, params_folder=None):
    max_path_length = 100
    batch_size = 16
    meta_batch_size = 1
    reward_arch = relu_net
    if reward_arch == relu_net:
        layers = 2
        d_hidden = 32
        reward_arch_args = {'layers':layers,
                            'd_hidden':d_hidden,}
    else:
        layers, d_hidden = 0, 0
        reward_arch_args = None

    # tf.reset_default_graph()
    env = TfEnv(CustomGymEnv('PointMazeRight-v0', record_video=False, record_log=False))
    barrier_range = [0.2, 0.6]
    barrier_y = 0.3

    # load ~2 iterations worth of data from each forward RL experiment as demos
    experts = load_latest_experts_multiple_runs('/atlas/u/lantaoyu/projects/InfoAIRL/data/maze_left_data_collect', n=4, latent_dim=latent_dim)

    irl_itr_list = [2800]

    for irl_itr in irl_itr_list:
        # params_file = os.path.join(DATA_DIR, '%s/itr_%d.pkl' % (params_folder, irl_itr))
        params_file = os.path.join(DATA_DIR, 'itr_%d.pkl' % irl_itr)
        prior_params = load_prior_params(params_file)
        init_context_encoder_params = load_prior_params(params_file, 'context_params')
        
        # params_file = os.path.join(DATA_DIR, 'itr_%d.pkl' % (irl_itr-800))
        policy_prior_params = load_prior_params(params_file, 'policy_params')
        # policy_prior_params = None

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
                            reward_arch=reward_arch,
                            reward_arch_args=reward_arch_args,
                            context_encoder=context_encoder,
                            state_only=True,
                            max_path_length=max_path_length,
                            meta_batch_size=meta_batch_size,
                            latent_dim=latent_dim)

        savedir = 'data_fusion_discrete/visualize_reward_right-%s' % irl_itr
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            irl_model.context_encoder.set_param_values(init_context_encoder_params)
            policy.set_param_values(policy_prior_params)
            irl_model.set_params(prior_params)
            boundary_low = -0.1
            boundary_high = 0.6
            
            expert_obs, expert_acts, expert_contexts = irl_model.extract_paths(irl_model.expert_trajs,
                                                    keys=('observations', 'actions', 'contexts'), T=max_path_length)
            expert_trajs = np.concatenate((expert_obs, expert_acts), axis=-1) # num_experts x T x (state_dim + act_dim)

            grid_size = 0.005
            rescale = 1./grid_size

            for itr in range(100):
                expert_traj_batch, m_batch = irl_model.sample_batch(expert_trajs, expert_contexts, batch_size=1, 
                        warm_up=False, warm_up_idx = False)
                obs_batch = []
                num_y = 0
                for pos_y in np.arange(boundary_low, boundary_high, grid_size):
                    num_y += 1
                    num_x = 0
                    for pos_x in np.arange(boundary_low, boundary_high, grid_size):
                        num_x += 1
                        obs_batch.append([pos_x, pos_y, 0.])
                obs_batch = np.array(obs_batch).reshape([1,-1, max_path_length,3])
                expert_traj_batch = np.tile(np.reshape(expert_traj_batch, [1,1,max_path_length,-1]),[1,obs_batch.shape[1],1,1])
                reward = tf.get_default_session().run(irl_model.reward,
                                                  feed_dict={irl_model.expert_traj_var: expert_traj_batch,
                                                            irl_model.obs_t: obs_batch})
                score = reward[:,0]
                ax = sns.heatmap(score.reshape([num_x, num_y]), cmap="YlGnBu_r")
                ax.scatter((m_batch[0][0][0]-boundary_low)*rescale , (m_batch[0][0][1]-boundary_low)*rescale,marker='*',s=150,c='r',edgecolors='k',linewidths=0.5)
                ax.scatter((0.3-boundary_low + np.random.uniform(low=-0.05, high=0.05))*rescale , (0.-boundary_low + np.random.uniform(low=-0.05, high=0.05))*rescale,marker='o',s=120,c='white',linewidths=0.5,edgecolors='k')
                ax.plot([(barrier_range[0] - boundary_low)*rescale, (barrier_range[1] - boundary_low)*rescale], [(barrier_y - boundary_low)*rescale, (barrier_y - boundary_low)*rescale],
                    color='k', linewidth=10)
                ax.invert_yaxis()
                plt.axis('off')
                plt.savefig(savedir + '/%s.png' % itr)
                print('Save Itr', itr)
                plt.close()


if __name__ == "__main__":
    import os
    # params_folders = DATA_DIR #os.listdir(DATA_DIR)
    params_dict = {
        # 'params_folder': [params_folders[1]],
        'latent_dim': [3],
    }
    # results = run_sweep_parallel(main, params_dict, repeat=1)
    run_sweep_serial(main, params_dict, repeat=1)
