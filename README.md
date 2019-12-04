## Meta-Inverse Reinforcement Learning with Probabilistic Context Variables<br>
Lantao Yu*, Tianhe Yu*, Chelsea Finn, Stefano Ermon.<br>
The 33rd Conference on Neural Information Processing Systems. (NeurIPS 2019)<br>
[[Paper]](https://arxiv.org/pdf/1909.09314.pdf) [[Website]](https://sites.google.com/view/pemirl)

### Usage
Requirement: The rllab package used in this project is provided [here](https://github.com/ermongroup/MetaIRL/tree/master/rllab).

To get expert trajectories for downstream tasks:
```
python scripts/maze_data_collect.py
```

After getting expert trajectories, run Meta-Inverse RL to learn context dependent reward functions:
```
python scripts/maze_wall_meta_irl.py
```
We provided a pretrained IRL model [here](https://github.com/ermongroup/MetaIRL/tree/master/data_fusion_discrete/maze_wall_meta_irl_imitcoeff-0.01_infocoeff-0.1_mbs-50_bs-16_itr-20_preepoch-1000_entropy-1.0_RandomPol_Rew-2-32/2019_05_14_02_33_17_0), which will be loaded by the following codes by default.

To visualize the context-dependent reward function (Figure 2 in the paper):
```
python scripts/maze_visualize_reward.py
```

To use the context-dependent reward function to train a new policy under new dynamics:
```
python scripts/maze_wall_meta_irl_test.py
```