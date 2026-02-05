# Robust Policy Learning for Multi-UAV Collision Avoidance with Causal Feature Selection

This repository is the official implementation of "Robust Policy Learning for Multi-UAV Collision Avoidance with Causal Feature Selection". It is designed for Multi-UAV Collision Avoidance .

## 🔥 Highlights

- We identify spurious correlations in visual representations as a key cause of poor generalization in DRL-based multi-UAV collision avoidance.
- A plug-and-play **Causal Feature Selection (CFS)** module is proposed to explicitly filter non-causal visual features during policy learning.
- We formulate a structural causal model for representation learning and provide a causal identifiability analysis to justify feature disentanglement.
- Hierarchical consistency constraints across representation, action, and Q-value levels guide the discovery of causal features without extra supervision.
- Extensive experiments on unseen backgrounds and obstacles show substantial improvements over SOTA methods in swarm and individual success rates.

## 📋 Overview

Collision avoidance navigation for unmanned aerial vehicle (UAV) swarms in complex and unseen outdoor environments presents a significant challenge, as UAVs are required navigate through various obstacles and intricate backgrounds. While existing deep reinforcement learning (DRL)-based collision avoidance methods have shown promising performance, they often suffer from poor generalization, leading to degraded performance in unseen environments. To address this limitation, we investigate the root causes of weak generalization in DRL models and propose a novel causal feature selection module. This module can be integrated into the policy network to effectively filter out non-causal factors in representations, thereby minimizing the impact of spurious correlations between non-causal elements and action predictions. Experimental results demonstrate that the proposed method achieves robust navigation performance and effective collision avoidance, particularly in scenarios with unseen backgrounds and obstacles, which significantly outperforms state-of-the-art (SOTA) algorithms.

### 🛩️ Demo of Multi-UAV Collision Avoidance in Unseen Scenarios

![Forest Obstacle Avoidance of Third-Person View](https://github.com/Gaofei-Han/CFS/blob/main/Forest%20Obstacle%20Avoidance%20(Third-Person%20View).gif)

![Canyon Obstacle Avoidance of Third-Person View](https://github.com/Gaofei-Han/CFS/blob/main/Canyon%20Obstacle%20Avoidance%20(Third-Person%20View).gif)

## Citation
```
@inproceedings{zhuang2025robust,
  title={Robust Policy Learning for Multi-UAV Collision Avoidance with Causal Feature Selection},
  author={Zhuang, Jiafan and Han, Gaofei and Xia, Zihao and Lin, Che and Wang, Boxi and Wang, Dongliang and   Li, Wenji and Hao, Zhifeng and Cai, Ruichu and Fan, Zhun},
  booktitle={Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems},
  pages={2392--2401},
  year={2025}
}
```