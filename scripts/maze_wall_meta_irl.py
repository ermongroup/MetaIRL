import tensorflow as tf

from inverse_rl.algos.meta_irl_trpo import MetaIRLTRPO
from inverse_rl.models.info_airl_state_train import *
from inverse_rl.models.tf_util import get_session_config
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.base import EnvSpec
from sandbox.rocky.tf.spaces.box import Box
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from inverse_rl.models.pretrain import *

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts, load_latest_experts_multiple_runs
from inverse_rl.models.architectures import relu_net, linear_net


def main(exp_name=None, fusion=False, latent_dim=3):
    max_path_length = 100
    info_coeff = 0.1
    imitation_coeff = 0.01
    batch_size = 16
    meta_batch_size = 50
    max_itrs = 20
    pre_epoch = 1000
    entropy_weight = 1.0
    reward_arch = relu_net
    if reward_arch == relu_net:
        layers = 2
        d_hidden = 32
        reward_arch_args = {'layers':layers,
                            'd_hidden':d_hidden,}
    else:
        layers, d_hidden = 0, 0
        reward_arch_args = None

    tf.reset_default_graph()
    env = TfEnv(CustomGymEnv('PointMazeLeft-v0', record_video=False, record_log=False))

    # load ~2 iterations worth of data from each forward RL experiment as demos
    experts = load_latest_experts_multiple_runs('data/maze_left_data_collect_discrete-15', n=4, latent_dim=latent_dim)

    # contexual policy pi(a|s,m)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

    # approximate posterior q(m|tau)
    context_encoder_spec = EnvSpec(
        observation_space=Box(np.tile(np.concatenate((env.observation_space.low[:-latent_dim], env.action_space.low)), max_path_length),
                                np.tile(np.concatenate((env.observation_space.high[:-latent_dim], env.action_space.high)), max_path_length)),
        action_space=Box(np.zeros(latent_dim), np.ones(latent_dim)),
    )
    context_encoder = GaussianMLPPolicy(name='context_encoder', env_spec=context_encoder_spec, hidden_sizes=(128, 128))

    pretrain_model = Pretrain(experts, policy, context_encoder, env, latent_dim, batch_size=400, kl_weight=0.1, epoch=pre_epoch)
    # pretrain_model = None
    if pretrain_model is None:
        pre_epoch = 0

    irl_model = InfoAIRL(env=env, policy=policy,
                        context_encoder=context_encoder,
                        reward_arch=reward_arch,
                        reward_arch_args=reward_arch_args,
                        expert_trajs=experts,
                        state_only=True,
                        max_path_length=max_path_length,
                        fusion=fusion,
                        max_itrs=max_itrs,
                        meta_batch_size=meta_batch_size,
                        imitation_coeff=imitation_coeff,
                        info_coeff=info_coeff,
                        latent_dim=latent_dim)

    algo = MetaIRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        randomize_policy=True,
        pretrain_model = pretrain_model,
        n_itr=3000,
        meta_batch_size=meta_batch_size,
        batch_size=batch_size,
        max_path_length=max_path_length,
        discount=0.99,
        store_paths=True,
        train_irl=True,
        irl_model_wt=1.0,
        entropy_weight=entropy_weight,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
    )
    if fusion:
        dirname = 'data_fusion_discrete_new/maze_wall_meta_irl_imitcoeff-%s_infocoeff-%s_mbs-%s_bs-%s_itr-%s_preepoch-%s_entropy-%s_RandomPol_Rew-%s-%s/%s' % (imitation_coeff, info_coeff, meta_batch_size, batch_size, max_itrs, pre_epoch, entropy_weight, layers, d_hidden, exp_name)
    else:
        dirname = 'data_discrete_new/maze_wall_meta_irl_imitcoeff-%s_infocoeff-%s_mbs-%s_bs-%s_itr-%s_preepoch-%s_entropy-%s_RandomPol_Rew-%s-%s/%s' % (imitation_coeff, info_coeff, meta_batch_size, batch_size, max_itrs, pre_epoch, entropy_weight, layers, d_hidden, exp_name)

    with rllab_logdir(algo=algo, dirname=dirname):
        with tf.Session():
            algo.train()


if __name__ == "__main__":
    params_dict = {
        'fusion': [True],
        'latent_dim': [3],
    }
    # run_sweep_parallel(main, params_dict, repeat=1)
    run_sweep_serial(main, params_dict, repeat=1)
