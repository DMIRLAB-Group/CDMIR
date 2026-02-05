# Learning Robust Policy for Multi-UAV Collision Avoidance via Compact Causal Feature

This repository is the official implementation of "Learning Robust Policy for Multi-UAV Collision Avoidance via Compact Causal Feature". It is designed for Multi-UAV Collision Avoidance .

## 🔥 Highlights

- We analyze the limitations of existing deep reinforcement learning methods in adapting to unseen environments and demonstrate the necessity of learning compact causal representations to enhance generalization.
- We propose a novel framework, Compact Causal Feature Learning, which extracts causal features from data while eliminating redundancy, resulting in representations that are both generalizable and efficient.
- We build a high-fidelity UAV simulation environment supporting causal feature learning and controlled interventions, enabling systematic evaluation under domain shifts. Experiments within this environment show that CCFL consistently outperforms state-of-the-art methods, significantly improving robustness and generalization in unseen scenarios.

## 📋 Overview

Deep reinforcement learning (DRL)-based multi-UAV collision avoidance methods often exhibit limited generalization when deployed in unseen environments, primarily due to the reliance on non causal and redundant visual features. Such overfitting to spurious correlations compromises both robustness and safety during real world deployment. To address these limitations, this study proposes a novel Compact Causal Feature Learning (CCFL) framework that enables UAVs to learn compact and generalizable causal representations. Specifically, a Causal Feature Identification module is designed to disentangle input representations into causal and non causal components, ensuring that the learned features preserve true environmental causality. Furthermore, a Redundancy Feature Compression module is introduced to remove redundant dependencies and compact the causal subspace, thereby enhancing generalization to previously unseen scenarios. Extensive experiments on a challenging UAV collision avoidance benchmark demonstrate that CCFL achieves substantial performance gains over state-of-the-art baselines, increasing individual success rates by 42.0% and swarm success rates by 61.6%. These results validate the effectiveness of compact causal feature learning for improving the adaptability, robustness, and safety of autonomous UAV systems operating in complex dynamic environments.

### 🛩️ Demo of Multi-UAV Collision Avoidance in Unseen Scenarios

![Forest Obstacle Avoidance of First-Person View](https://github.com/Gaofei-Han/CCFL/blob/main/Forest%20Obstacle%20Avoidance%20of%20First-Person%20View.gif)

![Canyon Obstacle Avoidance of Third-Person View](https://github.com/Gaofei-Han/CCFL/blob/main/Forest%20Obstacle%20Avoidance%20of%20Third-Person%20View.gif)

## Citation
```
@inproceedings{fan2026learning,
      title={Learning Robust Policy for Multi-UAV Collision Avoidance via Compact Causal Feature}, 
      author={Zhun Fan and Gaofei Han and Che Lin and Wenji Li and Jie Xu and Jiafan Zhuang },
      year={2026},
      booktitle={Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems}
}
```