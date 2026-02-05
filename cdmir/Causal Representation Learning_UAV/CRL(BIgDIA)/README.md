# Deep Reinforcement Learning based Multi-UAV Collision Avoidance with Causal Representation Learning

This repository is the official implementation of "Deep Reinforcement Learning based Multi-UAV Collision Avoidance with Causal Representation Learning". It is designed for Multi-UAV Collision Avoidance .

## 🔥 Highlights

- The study discovers that generalization failures in multi-UAV collision avoidance are primarily due to deep reinforcement learning (DRL) models overfitting to specific obstacle shapes encountered during training.
- We proposes a causal representation learning framework that extracts invariant features via obstacle-shape interventions.
- Extensive testing against state-of-the-art (SOTA) methods—including AutoAugment and DrAC—demonstrates that CRL achieves higher navigation success rates and better path efficiency (SPL) across unseen obstacles like triangles, pentagons, and cuboids.

## 📋 Overview

The deep reinforcement learning-based multi-UAV collision avoidance and navigation methods have made signif icant progress. However, the fundamental challenge of those methods is their restricted capability to generalize beyond the specific scenario in which they are trained on. We find that the cause of the generalization failures is attributed to spurious correlation. To solve this generalization problem, we propose a causal representation learning method to identify the causal representations from images. Specifically, our method can neglect factors of variation that are irrelevant to the deep reinforcement learning task through causal intervention. Subsequently, the causal representations are fed into the policy network for action prediction. Extensive testing reveals that our proposed method exhibits better generalization results compared to state-of-the-art method in different testing scenes.

## Citation
```
@inproceedings{han2024deep,
  title={Deep Reinforcement Learning Based Multi-UAV Collision Avoidance with Causal Representation       Learning},
  author={Han, Gaofei and Wu, Qingling and Wang, Boxi and Lin, Che and Zhuang, Jiafan and Li, Wenji and Hao, Zhifeng and Fan, Zhun},
  booktitle={2024 10th International Conference on Big Data and Information Analytics (BigDIA)},
  pages={833--839},
  year={2024},
  organization={IEEE}
}
```