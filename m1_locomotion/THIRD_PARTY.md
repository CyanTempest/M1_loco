# 来源与许可

M1 原始 URDF/STL 由用户提供，仅在本地项目内用于本次迁移；未对厂商资产声明新的开源许可。对外发布前需按原始权利归属处理。

本项目通过 Python 继承复用 `fan-ziqi/robot_lab` v2.3.2；上游为 Apache-2.0，许可证副本见 `docs/robot_lab_LICENSE`。Isaac Lab 上游为 BSD-3-Clause，Isaac Sim 按 NVIDIA 自身许可分发；二者的运行时未打包进本项目。

A2 腿部增益取自宇树官方 `unitreerobotics/unitree_rl_mjlab` commit `1425b15f73bd4095f0df53709d7c389c3eb9e790` 的 `src/assets/robots/unitree_a2/a2_constants.py`，并与该版本 `deploy/robots/a2/config/policy/velocity/v0/params/deploy.yaml` 交叉核对。本项目只记录参数事实和来源，不打包 A2 资产或策略。
