# M1 / UniLab：完整 B2W 风格训练

`b2w-full` 已接入 B2W 的 17 项非零奖励，以及材质、质量与惯量、质心、增益、倒地初始化、持续外力和间歇推扰，并增加每电机物理步延迟。M1 的 URDF、网格、标称质量/惯量、几何和力矩上限来自原始 CAD。

本目录运行：

```bash
bash train_mac_full.sh
```

脚本指定 **512 环境、30000 次 PPO 更新、六类地形、Apple MPS 网络训练、CPU MuJoCo 仿真**，使用已安装的环境，并在 macOS 上防止空闲休眠。

**30000 次正式训练由用户手动启动。** 新配置可从零训练；旧 basic 权重与新网络不兼容。你已运行的 full 配置在第170轮附近发生物理数值失稳，现已修复，保留的第1000号权重可通过以下命令恢复：

```bash
bash resume_mac_full.sh
```

脚本加载 `checkpoints/recovery_1000.pt`，追加28998次更新，与该权重已完成的1002次更新合计30000次。采用完整implicit积分、保持接触刚度的恢复系数映射、随机出生地形净空修正；所有17项奖励和原随机化类别保持启用。物理1kHz/策略50Hz。[失稳定位与修复](stability_fix.md) · [修复后验证记录](stability_fix_validation.json)

原64轮验收和约26小时估算属于修复前配置，已被本次验证替代；短验证不代表策略已经收敛。

修复后通过48项测试、512环境原生检查、512环境300次PPO更新及原checkpoint恢复验证，无数值失稳警告。300次更新约21.6分钟，线性外推30000次约36小时；持续发热和后台负载会影响耗时。

- [完整训练参数、启动与恢复说明](full_training.md)
- [任务 YAML](m1_unilab/task_full.yaml) · [地形 YAML](m1_unilab/terrain.yaml)
- [旧 basic 配置与视频](basic_profile.md) · [旧训练结果](results.md)
- [M1 模型来源核对](model_identity.md)

换电脑时保留 `m1_unilab` 与 `m1_locomotion` 两目录并排，先执行 `bash setup_mac.sh`。无需安装 Isaac Sim。固定 UniLab v1.5.9 / commit `f7f72afead492c4cbfcc64333bee39509c9a2204`、UniSim 2.7.0、MuJoCo 3.11.0、mjbatch-uni 0.2.5、Torch 2.14.1、RSL-RL 5.5.2；未修改上游核心。

当前碰撞器已改为libccd，绕开2026-11-11第1046轮的原生CCD段错误。详见 [碰撞器修复与续训](native_collision_fix.md)。旧300轮验收属于切换前配置。
