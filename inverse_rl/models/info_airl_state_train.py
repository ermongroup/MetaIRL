import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.spaces.box import Box

from inverse_rl.models.fusion_manager import RamFusionDistr, RamFusionDistrCustom
from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.architectures import relu_net
from inverse_rl.utils import TrainingIterator



class InfoAIRL(SingleTimestepIRL):
    """ 


    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """
    def __init__(self, env, policy,
                 context_encoder,
                 context_encoder_recurrent=False,
                 expert_trajs=None,
                 reward_arch=relu_net,
                 reward_arch_args=None,
                 value_fn_arch=relu_net,
                 score_discrim=False,
                 discount=1.0,
                 state_only=True,
                 max_path_length=500,
                 meta_batch_size=16,
                 max_itrs=100,
                 fusion=False,
                 latent_dim=3,
                 imitation_coeff=1.0,
                 info_coeff=1.0,
                 name='info_airl'):
        super(InfoAIRL, self).__init__()
        env_spec = env.spec
        if reward_arch_args is None:
            reward_arch_args = {}

        if fusion:
            self.fusion = RamFusionDistrCustom(100, subsample_ratio=0.5)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim - latent_dim
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env.action_space, Box)
        self.context_encoder = context_encoder
        self.score_discrim = score_discrim
        self.gamma = discount
        assert value_fn_arch is not None
        self.set_demos(expert_trajs)
        self.state_only = state_only
        self.T = max_path_length
        self.max_itrs = max_itrs
        self.latent_dim = latent_dim
        self.meta_batch_size = meta_batch_size
        self.policy = policy

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be meta_batch_size x batch_size x T x dO/dU
            self.expert_traj_var = tf.placeholder(tf.float32, [meta_batch_size, None, self.T, self.dO+self.dU], name='expert_traj')
            self.sample_traj_var = tf.placeholder(tf.float32, [meta_batch_size, None, self.T, self.dO+self.dU], name='sample_traj')
            self.obs_t = tf.placeholder(tf.float32, [meta_batch_size, None, self.T, self.dO], name='obs')
            self.nobs_t = tf.placeholder(tf.float32, [meta_batch_size, None, self.T, self.dO], name='nobs')
            self.act_t = tf.placeholder(tf.float32, [meta_batch_size, None, self.T, self.dU], name='act')
            self.nact_t = tf.placeholder(tf.float32, [meta_batch_size, None, self.T, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [meta_batch_size, None, 1, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [meta_batch_size, None, self.T, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            self.imitation_expert_obses = tf.placeholder(tf.float32, [meta_batch_size, None, self.T, self.dO]) # None = 1
            self.imitation_expert_acts = tf.placeholder(tf.float32, [meta_batch_size, None, self.T, self.dU])
            imitation_expert_obses = tf.reshape(self.imitation_expert_obses, [-1, self.dO])
            imitation_expert_acts = tf.reshape(self.imitation_expert_acts, [-1, self.dU])


            with tf.variable_scope('discrim') as dvs:
                # infer m_hat
                expert_traj_var = tf.reshape(self.expert_traj_var, [-1, (self.dO+self.dU)*self.T])
                # m_hat should be of shape meta_batch_size x (batch_size*2) x T x latent_dim
                context_dist_info_vars = self.context_encoder.dist_info_sym(expert_traj_var)
                context_mean_var = context_dist_info_vars["mean"]
                context_log_std_var = context_dist_info_vars["log_std"]
                eps = tf.random.normal(shape=tf.shape(context_mean_var))
                reparam_latent = eps * tf.exp(context_log_std_var) + context_mean_var

                self.reparam_latent_tile = reparam_latent_tile = tf.tile(tf.expand_dims(reparam_latent, axis=1), [1, self.T, 1])

                # One shot imitation
                self.imitation_reparam_latent_tile = tf.reshape(
                    tf.reshape(self.reparam_latent_tile, [meta_batch_size, -1, self.T, latent_dim])[:, 0, :, :], [-1, latent_dim]
                    )
                concat_obses_batch = tf.concat([imitation_expert_obses, self.imitation_reparam_latent_tile], axis=1)
                policy_dist_info_vars = policy.dist_info_sym(obs_var=concat_obses_batch)
                policy_likelihood_loss = -tf.reduce_mean(
                    policy.distribution.log_likelihood_sym(imitation_expert_acts, policy_dist_info_vars))

                reparam_latent_tile = tf.reshape(reparam_latent_tile, [-1, latent_dim])

                rew_input = self.obs_t
                if not self.state_only:
                    rew_input = tf.concat([self.obs_t, self.act_t], axis=-1)
                # condition on inferred m
                rew_input = tf.concat([tf.reshape(rew_input, [-1, rew_input.get_shape().dims[-1].value]), reparam_latent_tile], axis=1)
                with tf.variable_scope('reward'):
                    self.reward = reward_arch(rew_input, dout=1, **reward_arch_args)
                    self.sampled_traj_return = tf.reduce_sum(tf.reshape(self.reward, [meta_batch_size, -1, self.T]), axis=-1, keepdims=True)
                # with tf.variable_scope('reward', reuse=True):
                #     self.sampled_traj_return = reward_arch(tf.reshape(self.sampled_traj_var, [-1, self.dO+self.dU]), dout=1, **reward_arch)
                #     self.sampled_traj_return = tf.reduce_sum(tf.reshape(self.sampled_traj_return, [meta_batch_size, -1, self.T]), axis=-1)
                    #energy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

                npotential_input = tf.concat([tf.reshape(self.nobs_t, [-1, self.dO]), reparam_latent_tile], axis=-1)
                potential_input = tf.concat([tf.reshape(self.obs_t, [-1, self.dO]), reparam_latent_tile], axis=-1)

                # value function shaping
                with tf.variable_scope('vfn'):
                    fitted_value_fn_n = value_fn_arch(npotential_input, dout=1)
                with tf.variable_scope('vfn', reuse=True):
                    self.value_fn = fitted_value_fn = value_fn_arch(potential_input, dout=1)

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma*fitted_value_fn_n
                log_p_tau = self.reward  + self.gamma*fitted_value_fn_n - fitted_value_fn

            log_q_tau = self.lprobs
            log_p_tau = tf.reshape(log_p_tau, [meta_batch_size, -1, self.T, 1])

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0) # [meta_batch_size, -1, self.T, 1]
            self.discrim_output = tf.exp(log_p_tau-log_pq)
            cent_loss = -tf.reduce_mean(self.labels*(log_p_tau-log_pq) + (1-self.labels)*(log_q_tau-log_pq))

            # compute mutual information loss
            # sampled_traj_var = tf.reshape(tf.concat([self.obs_t, self.act_t], axis=-1), [-1, (self.dO+self.dU)*self.T])
            log_q_m_tau = tf.reshape(self.context_encoder.distribution.log_likelihood_sym(reparam_latent, context_dist_info_vars), [meta_batch_size, -1, 1])
            # Used for computing gradient w.r.t. psi
            info_loss = - tf.reduce_mean(log_q_m_tau * (1 - tf.squeeze(self.labels, axis=-1))) /  tf.reduce_mean(1- self.labels)
            # Used for computing the gradient w.r.t. theta
            info_surr_loss = - tf.reduce_mean(
                (1 - tf.squeeze(self.labels, axis=-1)) * log_q_m_tau * self.sampled_traj_return - 
                (1 - tf.squeeze(self.labels, axis=-1)) * log_q_m_tau * tf.reduce_mean(self.sampled_traj_return*(1 - tf.squeeze(self.labels, axis=-1)), axis=1, keepdims=True) / tf.reduce_mean(1 - self.labels)
                ) /  tf.reduce_mean(1 - self.labels)

            self.loss = cent_loss + info_coeff*info_loss
            self.info_loss = info_loss
            self.policy_likelihood_loss = policy_likelihood_loss
            tot_loss = self.loss
            context_encoder_weights = self.context_encoder.get_params(trainable=True)
            # reward_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="reward")
            reward_weights = [i for i in tf.trainable_variables() if "reward" in i.name]
            # value_fn_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vfn")
            value_fn_weights = [i for i in tf.trainable_variables() if "vfn" in i.name]
            
            # self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(tot_loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads_and_vars_cent = optimizer.compute_gradients(cent_loss, var_list=reward_weights+value_fn_weights+context_encoder_weights)
            grads_and_vars_context = optimizer.compute_gradients(info_coeff*info_loss, var_list=context_encoder_weights)
            grads_and_vars_reward = optimizer.compute_gradients(info_coeff*info_surr_loss, var_list=reward_weights)
            grads_and_vars_policy = optimizer.compute_gradients(imitation_coeff*policy_likelihood_loss, var_list=self.policy.get_params(trainable=True)+context_encoder_weights)
            self.step = optimizer.apply_gradients(grads_and_vars_cent+grads_and_vars_context+grads_and_vars_reward+grads_and_vars_policy)

            # grads_and_vars_cent = optimizer.compute_gradients(cent_loss, var_list=reward_weights+value_fn_weights)
            # grads_and_vars_reward = optimizer.compute_gradients(info_coeff*info_surr_loss, var_list=reward_weights)
            # self.step = optimizer.apply_gradients(grads_and_vars_cent+grads_and_vars_reward)
            
            self._make_param_ops(_vs)

    def fit(self, paths, expert_traj_batch=None, policy=None, batch_size=32, logger=None, lr=1e-3,**kwargs):
        meta_batch_size = self.meta_batch_size
        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(expert_traj_batch, n=len(paths[0]))
            self.fusion.add_paths(paths, expert_traj_batch, subsample=True)
            if old_paths is not None:
                for key in paths.keys():
                    paths[key] += old_paths[key]

        # Do we need to recalculate path probabilities every iteration since context encoderis being updated?
        # eval samples under current policy
        # TODO: fix this with dict
        self._compute_path_probs_dict(paths, insert=True)

        self._insert_next_state(paths)
        self._insert_next_state(self.expert_trajs)

        obs, obs_next, acts, acts_next, path_probs = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'), T=self.T)
        # TODO: we may need to assume that expert_trajs is also a dict
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_contexts = \
            self.extract_paths(self.expert_trajs,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'contexts'), T=self.T)

        # eval expert log probs under current policy
        expert_trajs = np.concatenate([expert_obs, expert_acts], axis=-1)
        m_hat_expert = self.context_encoder.get_actions(expert_trajs.reshape(-1, self.T*(self.dO+self.dU)))[0]
        self.eval_expert_probs(self.expert_trajs, policy, insert=True, context=m_hat_expert)

        expert_probs = self.extract_paths(self.expert_trajs, keys=('a_logprobs', ), T=self.T)[0]

        # Train discriminator
        expert_traj_batch_tile = np.tile(expert_traj_batch.reshape(meta_batch_size, 1, self.T, -1), [1, batch_size, 1, 1])

        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            # TODO: implement sample_batch in imitation_learning.py
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)
            if obs_batch.shape[-1] == self.dO + self.latent_dim:
                nobs_batch = nobs_batch[..., :-self.latent_dim]
                obs_batch = obs_batch[..., :-self.latent_dim]

            # First half of the batch is used for inferring m_hat
            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, batch_size=self.meta_batch_size*batch_size)
            if expert_obs_batch.shape[-1] == self.dO + self.latent_dim:
                nexpert_obs_batch = nexpert_obs_batch[..., :-self.latent_dim]
                expert_obs_batch = expert_obs_batch[..., :-self.latent_dim]

            # Build feed dict
            labels = np.zeros((meta_batch_size, batch_size*2, 1, 1))
            labels[:, batch_size:, ...] = 1.0
            imitation_expert_obses_input = expert_traj_batch.reshape(meta_batch_size, 1, self.T, -1)[:, :, :, :self.dO]
            imitation_expert_acts_input = expert_traj_batch.reshape(meta_batch_size, 1, self.T, -1)[:, :, :, self.dO:]
            expert_traj_batch_input = np.concatenate([expert_traj_batch_tile, np.concatenate([expert_obs_batch, expert_act_batch], axis=-1).reshape(meta_batch_size, batch_size, self.T, -1)], axis=1)
            sample_traj_batch = np.concatenate([obs_batch, act_batch], axis=-1)
            obs_batch = np.concatenate([obs_batch, expert_obs_batch.reshape(meta_batch_size, batch_size, self.T, -1)], axis=1)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch.reshape(meta_batch_size, batch_size, self.T, -1)], axis=1)
            act_batch = np.concatenate([act_batch, expert_act_batch.reshape(meta_batch_size, batch_size, self.T, -1)], axis=1)
            nact_batch = np.concatenate([nact_batch, nexpert_act_batch.reshape(meta_batch_size, batch_size, self.T, -1)], axis=1)
            lprobs_batch = np.concatenate([lprobs_batch, expert_lprobs_batch.reshape(meta_batch_size, batch_size, self.T, -1)], axis=1).astype(np.float32)
            feed_dict = {
                self.expert_traj_var: expert_traj_batch_input,
                self.sample_traj_var: sample_traj_batch,
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.nact_t: nact_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.imitation_expert_obses: imitation_expert_obses_input,
                self.imitation_expert_acts: imitation_expert_acts_input,
                self.lr: lr
                }

            loss, _ = tf.get_default_session().run([self.loss, self.step], feed_dict=feed_dict)
            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)

        if logger:
            logger.record_tabular('GCLDiscrimLoss', mean_loss)
            #obs_next = np.r_[obs_next, np.expand_dims(obs_next[-1], axis=0)]
            # TODO: fix this
            expert_traj_logging = np.tile(expert_traj_batch.reshape(meta_batch_size, 1, self.T, -1), [1, acts.shape[1], 1, 1])
            imitation_expert_obses_input = expert_traj_batch.reshape(meta_batch_size, 1, self.T, -1)[:, :, :, :self.dO]
            imitation_expert_acts_input = expert_traj_batch.reshape(meta_batch_size, 1, self.T, -1)[:, :, :, self.dO:]

            energy, logZ, dtau, info_loss, imit_loss = tf.get_default_session().run([self.reward, self.value_fn, self.discrim_output, self.info_loss, self.policy_likelihood_loss],
                                                               feed_dict={self.expert_traj_var: expert_traj_logging,
                                                                    self.sample_traj_var: np.concatenate([obs[..., :-self.latent_dim], acts], axis=-1),
                                                                    self.act_t: acts, self.obs_t: obs[..., :-self.latent_dim], self.nobs_t: obs_next[..., :-self.latent_dim],
                                                                    self.nact_t: acts_next,
                                                                    self.imitation_expert_obses: imitation_expert_obses_input,
                                                                    self.imitation_expert_acts: imitation_expert_acts_input,
                                                                    self.labels: np.zeros([meta_batch_size, acts.shape[1], 1, 1]),
                                                                    self.lprobs: path_probs})
            energy = -energy
            logger.record_tabular('GCLLogZ', np.mean(logZ))
            logger.record_tabular('GCLAverageEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageLogPtau', np.mean(-energy-logZ))
            logger.record_tabular('GCLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('GCLMedianLogQtau', np.median(path_probs))
            logger.record_tabular('GCLAverageDtau', np.mean(dtau))
            logger.record_tabular('GCLAverageMutualInfo', np.mean(info_loss))
            logger.record_tabular('GCLAverageImitationLoss', np.mean(imit_loss))


            #expert_obs_next = np.r_[expert_obs_next, np.expand_dims(expert_obs_next[-1], axis=0)]
            # Not sure if using expert trajectories for expert_traj_var and sample_traj_var makes sense
            # energy, logZ, dtau, info_loss = tf.get_default_session().run([self.reward, self.value_fn, self.discrim_output, self.info_loss],
            #         feed_dict={self.expert_traj_var: np.concatenate([expert_obs, expert_acts], axis=-1),
            #                     self.sample_traj_var: np.concatenate([expert_obs, expert_acts], axis=-1),
            #                         self.act_t: expert_acts, self.obs_t: expert_obs, self.nobs_t: expert_obs_next,
            #                         self.nact_t: expert_acts_next,
            #                         self.labels: np.zeros([meta_batch_size, acts.shape[1], 1, 1]),
            #                         self.lprobs: expert_probs})
            # energy = -energy
            # logger.record_tabular('GCLAverageExpertEnergy', np.mean(energy))
            # logger.record_tabular('GCLAverageExpertLogPtau', np.mean(-energy-logZ))
            # logger.record_tabular('GCLAverageExpertLogQtau', np.mean(expert_probs))
            # logger.record_tabular('GCLMedianExpertLogQtau', np.median(expert_probs))
            # logger.record_tabular('GCLAverageExpertDtau', np.mean(dtau))
            # logger.record_tabular('GCLAverageExpertMutualInfo', np.mean(info_loss))
        return mean_loss

    def eval(self, paths, expert_traj_batch=None, **kwargs):
        """
        Return bonus
        """
        # TODO: finish this
        if self.score_discrim:
            self._compute_path_probs_dict(paths, insert=True)
            obs, obs_next, acts, path_probs = self.extract_paths(paths, keys=('observations', 'observations_next', 'actions', 'a_logprobs'), T=self.T)
            expert_traj_batch = np.tile(expert_traj_batch.reshape(self.meta_batch_size, 1, self.T, -1), [1, acts.shape[1], 1, 1])
            path_probs = np.expand_dims(path_probs, axis=2)
            scores = tf.get_default_session().run(self.discrim_output,
                                              feed_dict={self.expert_traj_var: expert_traj_batch,
                                                         self.act_t: acts, self.obs_t: obs[..., :-self.latent_dim],
                                                         self.nobs_t: obs_next[..., :-self.latent_dim],
                                                         self.lprobs: path_probs})
            score = np.log(scores) - np.log(1-scores)
            score = score[:,0]
        else:
            obs, acts = self.extract_paths(paths, T=self.T)
            expert_traj_batch = np.tile(expert_traj_batch.reshape(self.meta_batch_size, 1, self.T, -1), [1, acts.shape[1], 1, 1])
            reward = tf.get_default_session().run(self.reward,
                                              feed_dict={self.expert_traj_var: expert_traj_batch,
                                                        self.act_t: acts, self.obs_t: obs[..., :-self.latent_dim]})
            score = reward[:,0]
        score = score.reshape(self.meta_batch_size, -1, self.T)
        return self.unpack(score, paths)

    def eval_single(self, obs):
        reward = tf.get_default_session().run(self.reward,
                                              feed_dict={self.obs_t: obs})
        score = reward[:, 0]
        return score

    def debug_eval(self, paths, **kwargs):
        obs, acts = self.extract_paths(paths, T=self.T)
        reward, v, qfn = tf.get_default_session().run([self.reward, self.value_fn,
                                                            self.qfn],
                                                      feed_dict={self.act_t: acts, self.obs_t: obs})
        return {
            'reward': reward,
            'value': v,
            'qfn': qfn,
        }

