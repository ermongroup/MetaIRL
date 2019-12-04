from inverse_rl.envs.env_utils import *
from sandbox.rocky.tf.envs.base import TfEnv
from inverse_rl.utils.log_utils import *
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.base import EnvSpec
from sandbox.rocky.tf.spaces.box import Box
from inverse_rl.envs.env_utils import CustomGymEnv

class Pretrain(object):
	def __init__(self, experts, policy, context_encoder, env, latent_dim, batch_size, kl_weight, epoch=3000):
		flat_experts = []
		expert_obses = []
		expert_actions = []
		expert_latent = []
		state_dim = len(experts[0]["observations"][0])
		action_dim = len(experts[0]["actions"][0])
		self.expert = experts
		self.policy = policy
		self.context_encoder = context_encoder
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.env = env
		self.epoch = epoch

		self.max_path_length = max_path_length = len(experts[0]["actions"])

		for i in range(len(experts)):
			traj = np.concatenate([experts[i]["observations"], experts[i]["actions"]], axis=1)
			flat_experts.append(np.reshape(traj, [-1]))
			expert_obses.append(experts[i]["observations"])
			expert_actions.append(experts[i]["actions"])
			expert_latent.append(experts[i]["contexts"])

		self.num_batch = int(np.ceil(len(flat_experts) / batch_size))


		self.flat_traj_batch_input = tf.placeholder(tf.float32, shape=[None, (state_dim + action_dim) * max_path_length]) # [batch_size, latent_dim]
		self.obses_batch_input = tf.placeholder(tf.float32, shape=[None, state_dim]) # [batch_size * time, state_dim]
		self.actions_batch_input = tf.placeholder(tf.float32, shape=[None, action_dim]) # [batch_size * time, action_dim]
		self.lr = tf.placeholder(tf.float32)

		self.policy_dist_info_vars, self.reparam_latent, self.policy_likelihood_loss, self.latent_loss, self.total_loss = \
			self.build_graph(self.flat_traj_batch_input, self.obses_batch_input, self.actions_batch_input, kl_weight)

		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.train_op = optimizer.minimize(self.total_loss, var_list=[policy.get_params(trainable=True),
																 context_encoder.get_params(trainable=True)])
		self.flat_experts = flat_experts
		self.expert_obses = expert_obses
		self.expert_actions = expert_actions
		self.expert_latent = expert_latent


	def build_graph(self, flat_traj_batch_input, obses_batch_input, actions_batch_input, kl_weight, train=True):
		context_dist_info_vars = self.context_encoder.dist_info_sym(obs_var=flat_traj_batch_input)
		context_mean_var = context_dist_info_vars["mean"]
		context_log_std_var = context_dist_info_vars["log_std"]
		eps = tf.random.normal(shape=tf.shape(context_mean_var))
		reparam_latent = eps * tf.exp(context_log_std_var) + context_mean_var  # [batch_size, latent_dim]

		if train:
			reparam_latent_tile = tf.tile(tf.expand_dims(reparam_latent, axis=1), [1, self.max_path_length, 1])
			reparam_latent_tile = tf.reshape(reparam_latent_tile, [-1, self.latent_dim])
		else:
			reparam_latent_tile = reparam_latent

		# reparam_latent_tile = tf.zeros_like(reparam_latent_tile)
		concat_obses_batch = tf.concat([obses_batch_input, reparam_latent_tile], axis=1)
		policy_dist_info_vars = self.policy.dist_info_sym(obs_var=concat_obses_batch)
		policy_likelihood_loss = -tf.reduce_mean(
			self.policy.distribution.log_likelihood_sym(actions_batch_input, policy_dist_info_vars))

		log_pz = self.log_normal_pdf(reparam_latent, 0., 0.)
		log_qz = self.log_normal_pdf(reparam_latent, context_mean_var, context_log_std_var * 2.)
		latent_loss = tf.reduce_mean(log_qz - log_pz)
		total_loss = policy_likelihood_loss + kl_weight * latent_loss
		return policy_dist_info_vars, reparam_latent, policy_likelihood_loss, latent_loss, total_loss

	def run(self, sess):
		flat_experts = self.flat_experts
		expert_obses = self.expert_obses
		expert_actions = self.expert_actions
		expert_latent = self.expert_latent
		batch_size = self.batch_size
		print("================= Pretraining ==============")
		for e in range(self.epoch):
			loss1_list = []
			loss2_list = []
			total_loss_list = []
			latent_error = []
			for i in range(self.num_batch):
				flat_traj_batch = flat_experts[i * batch_size:(i + 1) * batch_size]
				obses_batch = np.array(expert_obses[i * batch_size:(i + 1) * batch_size]).reshape(
					[batch_size * self.max_path_length, -1])
				actions_batch = np.array(expert_actions[i * batch_size:(i + 1) * batch_size]).reshape(
					[batch_size * self.max_path_length, -1])
				latent_batch = np.array(expert_latent[i * batch_size:(i + 1) * batch_size])
				feed_dict = {self.flat_traj_batch_input: flat_traj_batch,
							 self.obses_batch_input: obses_batch,
							 self.actions_batch_input: actions_batch,
							 self.lr: 1e-3}
				# summary, _, loss1, loss2 = sess.run([merged, train_op, policy_likelihood_loss, latent_loss], feed_dict)
				# train_writer.add_summary(summary, i+e*num_batch)
				_, loss1, loss2, loss3 = sess.run([self.train_op, self.policy_likelihood_loss, self.latent_loss, self.total_loss], feed_dict)

				loss1_list.append(loss1)
				loss2_list.append(loss2)
				total_loss_list.append(loss3)
			if e % 100 == 0:
				print("Pretrain Epoch", e, "PolicyLikelihood", np.mean(loss1_list), "KL", np.mean(loss2_list), "Total", np.mean(total_loss_list))

		return self.policy, self.context_encoder


	def log_normal_pdf(self, sample, mean, logvar, raxis=1):
		log2pi = tf.math.log(2. * np.pi)
		return tf.reduce_sum(
			-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
			axis=raxis)
