from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from inverse_rl.algos.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from inverse_rl.algos.meta_irl_batch_polopt import MetaIRLBatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class MetaIRLNPO(MetaIRLBatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            entropy_weight=1.0,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(name='lbfgs')
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.pol_ent_wt = entropy_weight
        super(MetaIRLNPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        # lists over the meta_batch_size0
        context = tf.reshape(self.irl_model.reparam_latent_tile, [self.meta_batch_size, -1, self.irl_model.T, self.irl_model.latent_dim])

        # if not self.train_irl:
            # context = tf.stop_gradient(context)

        obs_vars = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_vars = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_vars = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        clean_obs_vars = tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + [self.env.observation_space.flat_dim - self.irl_model.latent_dim], 
                name='clean_obs'
        )
        policy_input = tf.reshape(tf.concat([tf.reshape(clean_obs_vars, [self.meta_batch_size, -1, self.irl_model.T, self.env.observation_space.flat_dim - self.irl_model.latent_dim]), 
                                                context], axis=-1), [-1, self.env.observation_space.flat_dim])

        # input_list = obs_vars + action_vars + advantage_vars
        input_list = [clean_obs_vars] + [action_vars] + [advantage_vars] + [self.irl_model.expert_traj_var]

        dist = self.policy.distribution

        old_dist_info_vars_list, state_info_vars_list = [], []
        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list += [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='%s' % k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list += [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_vars = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_vars = None

        surr_losses, mean_kls = [], []
        # dist_info_vars = self.policy.dist_info_sym(obs_vars[i], state_info_vars[i])
        dist_info_vars = self.policy.dist_info_sym(policy_input, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_vars, old_dist_info_vars, dist_info_vars)

        if self.pol_ent_wt > 0:
            if 'log_std' in dist_info_vars:
                log_std = dist_info_vars['log_std']
                ent = tf.reduce_sum(log_std + tf.log(tf.sqrt(2 * np.pi * np.e)), reduction_indices=-1)
            elif 'prob' in dist_info_vars:
                prob = dist_info_vars['prob']
                ent = -tf.reduce_sum(prob*tf.log(prob), reduction_indices=-1)
            else:
                raise NotImplementedError()
            ent = tf.stop_gradient(ent)
            adv = advantage_vars + self.pol_ent_wt*ent
        else:
            adv = advantage_vars

        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_vars) / tf.reduce_sum(valid_vars)
            surr_loss = - tf.reduce_sum(lr * adv * valid_vars) / tf.reduce_sum(valid_vars)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = - tf.reduce_mean(lr * adv)
        surr_losses.append(surr_loss)
        mean_kls.append(mean_kl)

        surr_loss = tf.reduce_mean(tf.stack(surr_losses, 0))  # mean over meta_batch_size (the diff tasks)
        mean_kl = tf.reduce_mean(tf.stack(mean_kls))
        input_list += state_info_vars_list + old_dist_info_vars_list

        if is_recurrent:
            input_list += valid_vars

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data, expert_traj_batch):
        input_list = []
        obs_list, action_list, adv_list = [], [], []
        for i in range(self.meta_batch_size):

            inputs = ext.extract(
                samples_data[i],
                "observations", "actions", "advantages"
            )
            obs_list.append(inputs[0][:, :-3])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])
        obs_list = [np.concatenate(obs_list, axis=0)]
        action_list = [np.concatenate(action_list, axis=0)]
        adv_list = [np.concatenate(adv_list, axis=0)]
        input_list += obs_list + action_list + adv_list + [np.tile(np.expand_dims(expert_traj_batch, axis=1), 
                                                            [1,samples_data[0]['observations'].shape[0]//self.irl_model.T,1,1])]
                                                            # [1,int(self.batch_size/self.max_path_length/self.meta_batch_size),1,1])]
        # all_input_values = tuple(ext.extract(
        #     samples_data,
        #     "observations", "actions", "advantages",
        # ))
        state_info_list, dist_info_list = {k:[] for k in self.policy.state_info_keys}, {k:[] for k in self.policy.distribution.dist_info_keys}
        for i in range(self.meta_batch_size):
            agent_infos = samples_data[i]["agent_infos"]
            for k in self.policy.state_info_keys:
                state_info_list[k].append(agent_infos[k])
            for k in self.policy.distribution.dist_info_keys:
                dist_info_list[k].append(agent_infos[k])
        state_info_list = {k: np.concatenate(state_info_list[k]) for k in self.policy.state_info_keys}
        dist_info_list = {k: np.concatenate(dist_info_list[k]) for k in self.policy.distribution.dist_info_keys}
        input_list += tuple(state_info_list.values()) + tuple(dist_info_list.values())
        if self.policy.recurrent:
            valids_list = []
            for i in range(self.meta_batch_size):
                valids_list.append(samples_data[i]["valids"])
            valids_list = np.concatenate(valids_list, axis=0)
            input_list += tuple(valids_list)

        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_list)
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(input_list)
        logger.log("Optimizing")
        self.optimizer.optimize(input_list)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(input_list)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_list)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    # @overrides
    # def init_opt(self):
    #     is_recurrent = int(self.policy.recurrent)
    #     # lists over the meta_batch_size0
    #     obs_vars, action_vars, advantage_vars = [], [], []
    #     clean_obs_vars = []
    #     policy_input = []
    #     context = tf.reshape(self.irl_model.reparam_latent_tile, [self.meta_batch_size, -1, self.irl_model.T, self.irl_model.latent_dim])
    #     for i in range(self.meta_batch_size):
    #         obs_vars.append(self.env.observation_space.new_tensor_variable(
    #             'obs' + '_' + str(i),
    #             extra_dims=1 + is_recurrent,
    #         ))
    #         action_vars.append(self.env.action_space.new_tensor_variable(
    #             'action' + '_' + str(i),
    #             extra_dims=1 + is_recurrent,
    #         ))
    #         advantage_vars.append(tensor_utils.new_tensor(
    #             'advantage' + '_' + str(i),
    #             ndim=1 + is_recurrent,
    #             dtype=tf.float32,
    #         ))
    #         clean_obs_vars.append(tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + [self.env.observation_space.flat_dim - self.irl_model.latent_dim], 
    #                 name='clean_obs' + '_' + str(i)
    #         ))
    #         policy_input.append(tf.reshape(tf.concat([tf.reshape(clean_obs_vars[i], [-1, self.irl_model.T, self.env.observation_space.flat_dim - self.irl_model.latent_dim]), 
    #                                                 context[i]], axis=-1), [-1, self.env.observation_space.flat_dim]))

    #     # input_list = obs_vars + action_vars + advantage_vars
    #     input_list = clean_obs_vars + action_vars + advantage_vars + [self.irl_model.expert_traj_var]

    #     dist = self.policy.distribution

    #     old_dist_info_vars, old_dist_info_vars_list, state_info_vars, state_info_vars_list = [], [], [], []
    #     for i in range(self.meta_batch_size):
    #         old_dist_info_vars.append({
    #             k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s_%s' % (i, k))
    #             for k, shape in dist.dist_info_specs
    #             })
    #         old_dist_info_vars_list += [old_dist_info_vars[i][k] for k in dist.dist_info_keys]

    #         state_info_vars.append({
    #             k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='%s_%s' % (k, i))
    #             for k, shape in self.policy.state_info_specs
    #             })
    #         state_info_vars_list += [state_info_vars[i][k] for k in self.policy.state_info_keys]

    #     if is_recurrent:
    #         valid_vars = [tf.placeholder(tf.float32, shape=[None, None], name="valid_%s" % i) for i in range(self.meta_batch_size)]
    #     else:
    #         valid_vars = [None for _ in range(self.meta_batch_size)]

    #     surr_losses, mean_kls = [], []
    #     for i in range(self.meta_batch_size):
    #         # dist_info_vars = self.policy.dist_info_sym(obs_vars[i], state_info_vars[i])
    #         dist_info_vars = self.policy.dist_info_sym(policy_input[i], state_info_vars[i])
    #         kl = dist.kl_sym(old_dist_info_vars[i], dist_info_vars)
    #         lr = dist.likelihood_ratio_sym(action_vars[i], old_dist_info_vars[i], dist_info_vars)

    #         if self.pol_ent_wt > 0:
    #             if 'log_std' in dist_info_vars:
    #                 log_std = dist_info_vars['log_std']
    #                 ent = tf.reduce_sum(log_std + tf.log(tf.sqrt(2 * np.pi * np.e)), reduction_indices=-1)
    #             elif 'prob' in dist_info_vars:
    #                 prob = dist_info_vars['prob']
    #                 ent = -tf.reduce_sum(prob*tf.log(prob), reduction_indices=-1)
    #             else:
    #                 raise NotImplementedError()
    #             ent = tf.stop_gradient(ent)
    #             adv = advantage_vars[i] + self.pol_ent_wt*ent
    #         else:
    #             adv = advantage_vars[i]

    #         if is_recurrent:
    #             mean_kl = tf.reduce_sum(kl * valid_vars[i]) / tf.reduce_sum(valid_vars[i])
    #             surr_loss = - tf.reduce_sum(lr * adv * valid_vars[i]) / tf.reduce_sum(valid_vars[i])
    #         else:
    #             mean_kl = tf.reduce_mean(kl)
    #             surr_loss = - tf.reduce_mean(lr * adv)
    #         surr_losses.append(surr_loss)
    #         mean_kls.append(mean_kl)

    #     surr_loss = tf.reduce_mean(tf.stack(surr_losses, 0))  # mean over meta_batch_size (the diff tasks)
    #     mean_kl = tf.reduce_mean(tf.stack(mean_kls))
    #     input_list += state_info_vars_list + old_dist_info_vars_list

    #     if is_recurrent:
    #         input_list += valid_vars

    #     self.optimizer.update_opt(
    #         loss=surr_loss,
    #         target=self.policy,
    #         leq_constraint=(mean_kl, self.step_size),
    #         inputs=input_list,
    #         constraint_name="mean_kl"
    #     )
    #     return dict()

    # @overrides
    # def optimize_policy(self, itr, samples_data, expert_traj_batch):
    #     input_list = []
    #     obs_list, action_list, adv_list = [], [], []
    #     for i in range(self.meta_batch_size):

    #         inputs = ext.extract(
    #             samples_data[i],
    #             "observations", "actions", "advantages"
    #         )
    #         obs_list.append(inputs[0][:, :-3])
    #         action_list.append(inputs[1])
    #         adv_list.append(inputs[2])
    #     input_list += obs_list + action_list + adv_list + [np.tile(np.expand_dims(expert_traj_batch, axis=1), 
    #                                                         [1,int(self.batch_size/self.max_path_length/self.meta_batch_size),1,1])]
    #     # all_input_values = tuple(ext.extract(
    #     #     samples_data,
    #     #     "observations", "actions", "advantages",
    #     # ))
    #     dist_info_list, state_info_list = [], []
    #     for i in range(self.meta_batch_size):
    #         agent_infos = samples_data[i]["agent_infos"]
    #         state_info_list += [agent_infos[k] for k in self.policy.state_info_keys]
    #         dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
    #     input_list += tuple(state_info_list) + tuple(dist_info_list)
    #     if self.policy.recurrent:
    #         valids_list = []
    #         for i in range(self.meta_batch_size):
    #             valids_list.append(samples_data[i]["valids"])
    #         input_list += tuple(valids_list)

    #     logger.log("Computing loss before")
    #     loss_before = self.optimizer.loss(input_list)
    #     logger.log("Computing KL before")
    #     mean_kl_before = self.optimizer.constraint_val(input_list)
    #     logger.log("Optimizing")
    #     self.optimizer.optimize(input_list)
    #     logger.log("Computing KL after")
    #     mean_kl = self.optimizer.constraint_val(input_list)
    #     logger.log("Computing loss after")
    #     loss_after = self.optimizer.loss(input_list)
    #     logger.record_tabular('LossBefore', loss_before)
    #     logger.record_tabular('LossAfter', loss_after)
    #     logger.record_tabular('MeanKLBefore', mean_kl_before)
    #     logger.record_tabular('MeanKL', mean_kl)
    #     logger.record_tabular('dLoss', loss_before - loss_after)
    #     return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            policy_params=self.policy.get_param_values(),
            context_params=self.irl_model.context_encoder.get_param_values(),
            irl_params=self.get_irl_params(),
            baseline=self.baseline,
            env=self.env,
        )
