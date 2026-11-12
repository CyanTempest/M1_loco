# M1 完整训练与迁移说明

## 启动与恢复

本机已安装，终端运行：

```bash
bash "/Users/amiao/Documents/Codex/2026-11-10/1-m1-urdf-nx-ota-calibration/outputs/m1_unilab/train_mac_full.sh"
```

等价训练入口：`python -m m1_unilab.train --profile b2w-full --terrain mixed --num-envs 512 --iterations 30000 --device mps --torch-threads 2`。脚本通过 `uv run` 使用本目录 `.venv`，或本次工作区 `../../work/unilab-venv`；`M1_UNILAB_ENV` 可覆盖。另一台 Mac 先运行 `bash setup_mac.sh`。

以上命令从零开始。此前视频来自 basic 旧权重，不代表新配置已收敛；旧53/243维、128/128/64网络的 checkpoint 不兼容。

用户第一次full训练在第170轮附近失稳，已修复。恢复保留下来的第1000号full权重，运行：

```bash
bash "/Users/amiao/Documents/Codex/2026-11-10/1-m1-urdf-nx-ota-calibration/outputs/m1_unilab/resume_mac_full.sh"
```

该权重已经完成1002次更新（按累计样本数核对），恢复脚本追加28998次，合计30000次。修复后正式长训练由用户手动启动。

日志、TensorBoard、run_config/run_summary、权重保存在本包 `runs/M1MixedFull/<时间>_mujoco/`。每100次更新保存；正常结束也保存最终权重。Ctrl+C可中断，恢复时使用已保存的权重。例如追加25000次更新：

```bash
bash train_mac_full.sh --checkpoint "/absolute/path/to/new_run/model_4999.pt" --iterations 25000
```

`--iterations` 是本次追加数。恢复模型/优化器后，仿真环境和随机数状态重新建立，不是整个物理轨迹逐比特续跑。

## 训练参数

| 项目 | 值 |
|---|---|
| 设备 | Apple M5 / MPS 网络；CPU 原生 MuJoCo/mjbatch 物理 |
| 环境 / 更新 | 512 / 30000 |
| Rollout | 每环境24策略步，每次12288条转移 |
| 计划总样本 | 368,640,000条环境转移，7,372,800,000个环境物理步 |
| 时步 | physics .001s，policy .02s，decimation20 |
| 积分器 | MuJoCo implicit；禁止内部自动重置掩盖数值故障 |
| Episode | 20s / 1000策略步 |
| Actor / critic | 512/256/128、ELU，观测运行归一化关闭 |
| 输入 / 输出 | actor57，critic247，action16 |
| PPO | epochs5，minibatches4，clip.2，value coef1，clipped value loss |
| 优化 | lr.001、adaptive KL.01、entropy.01、gamma.99、GAE lambda.95、max gradient1 |
| 探索 / 种子 | 初始高斯std1；seed1，terrain seed17 |

Actor包含角速度3、重力3、命令3、相对关节位置16（轮位清零后加噪声）、关节速度16、上一动作16。Critic另有线速度3和187点高度。Actor角速度scale.25、关节速度scale.05；缩放前均匀噪声分别为角速度±.2、重力±.05、位置±.01、关节速度±1.5。Critic无噪声，各项scale1，高度为 `clip(base_z-ground_z-.5,-1,1)`。

命令每10s更新：vx/vy各±1m/s，yaw rate±1rad/s；全部heading模式，目标±π、gain.5；2%站立，水平命令范数≤.2置零。不使用命令范围课程。

## 地形与课程

| 类型 | 环境权重 | 范围 |
|---|---:|---|
| 平地 | 20% | 平面 |
| 上坡 / 下坡 | 各10% | 坡度比0–.35，最大约19.3° |
| 粗糙 | 20% | −4至+4cm噪声，10cm噪声采样间距 |
| 上楼梯 / 下楼梯 | 各20% | 高3–16cm、踏面30cm、中心平台2m |

5行难度×6列类型，每块8×8m，边界4m；水平分辨率5cm、竖直量化5mm。配置上下界不保证恰好在随机地图取到；512环境的比例会因取整略有差异。平地、粗糙噪声不随等级变；坡度与台阶高度随等级增长。

首次从level0开始。回合结束，水平位移>4m升级；否则位移<当前平移命令速度×20s×.5降级。最低0，越过最高级随机进入有效级。仅20s timeout及整张地图边缘3m缓冲区终止，均按timeout bootstrap；无倾倒/接触终止，允许倒地恢复。边界为 `|x|>21m` 或 `|y|>25m`，不是每个tile边界。

这是M1四类地形方案：原B2W的10×20、5–23cm台阶、boxes20%并未原样复制；M1增加单独平地，当前无boxes。高度场楼梯是离散近似，不等同Isaac三角网格严格竖直台阶。

## 完整17项奖励

原始值乘下列权重，再由RewardManager乘policy dt=.02一次。`G=clip(-gravity_z,0,.7)/.7`。扭矩来自原生 `jointactuatorfrc`；加速度每物理步差分；接触力记录最近3个物理步。

| 项目 | 权重 | 定义 |
|---|---:|---|
| lin_vel_z_l2 | −2 | G×垂向速度平方 |
| ang_vel_xy_l2 | −.05 | G×roll/pitch角速度平方和 |
| joint_torques_l2 | −1e−5 | 12腿实际扭矩平方和 |
| joint_acc_l2 | −1e−7 | 12腿加速度平方和 |
| joint_acc_wheel_l2 | −2.5e−10 | 4轮加速度平方和 |
| joint_pos_limits | −5 | 12腿越过.9 soft limit的距离和 |
| joint_power | −1e−5 | 12腿sum(abs(torque*velocity))，按B2W不包含轮 |
| stand_still | −2 | cmd范数<.1时腿偏差L1×G |
| joint_pos_penalty | −1 | 腿偏差L2范数×G；cmd≤.1且水平速度≤.5时乘5 |
| joint_mirror | −.05 | 两组对角腿位置差平方和平均×G，使用M1几何坐标映射 |
| action_rate_l2 | −.01 | 16维动作一阶差分平方和 |
| undesired_contacts | −1 | 非轮body历史最大合力>1N的数量×G |
| contact_forces | −1.5e−4 | 四轮历史最大合力超过100N的超量和，不平方 |
| track_lin_vel_xy_exp | +3 | exp(−水平速度误差平方和/.25)×G |
| track_ang_vel_z_exp | +1.5 | exp(−yaw rate误差平方/.25)×G |
| feet_contact_without_cmd | +.1 | 无命令时四轮首次接触数量×G |
| upward | +3 | (1−gravity_z)²，直立原值4、倒立0 |

First contact按接触年龄 `(0,.02+1e−8)` 判定，不能替换成持续接触数量。M1前后腿绕z旋转π对应几何镜像，奖励中后腿ABAD/HIP/KNEE三轴取反后比较；不改变电机坐标。默认站姿镜像误差为0。

## 全部随机化

| 对象 | 时机 | 范围 |
|---|---|---|
| 材质 | startup | 64共享bucket，每环境/碰撞geom独立选择；μ .3–1，目标恢复系数0–.5 |
| 机身质量 | startup | M1基准加−1至+3kg |
| 其他body质量 | startup | 各M1基准乘.7–1.3 |
| 惯量 | 随质量startup | 主惯量按质量比同步缩放，保留主轴 |
| 机身质心 | startup | xyz各±5cm |
| 16电机kp/kd | 每reset | 相对标称各独立乘.5–2，轮kp恒0 |
| 16电机延迟 | 每reset | 独立整数0–20物理步，即0–20ms，整回合固定 |
| 持续外力/外扭矩 | 每reset | 机身局部xyz各±30N / ±10N·m，每物理子步重新施加 |
| 根部初始化 | 每reset | xy±.5m、z+0–.2m、RPY各±3.14；线/角速度各±.5 |
| 关节初始化 | 每reset | M1默认位置、零速度，对应B2W位置×[1,1]与速度[0,0] |
| 地形净空修正 | 根部随机化后 | 按全部55个碰撞几何和实际地形，仅在需要时抬升Z，保留5mm间隙 |
| 推扰 | 每10–15s | 世界系vx/vy各增加±.5m/s，保留其余速度 |

Startup值不随reset重采样；增益相对标称、不累乘。**电机延迟为用户要求的扩展，B2W原配置没有延迟**；0–20ms是暂定范围，不是sysid实测值。

净空修正避免随机倒地姿态出生时穿进楼梯，XY、朝向、线/角速度及随机数采样顺序保持不变。它使用包围球作保守估计，最终出生Z可能高于原始z+0–.2m采样；正常直立且已有足够净空时不抬升。该处理是针对M1与MuJoCo的适配，并非B2W原始重置代码。

## 控制器与迁移差异

12腿沿用A2增益：ABAD/HIP100/4、KNEE150/6；四轮采用用户选择的B2W kp0/kd1。腿目标q0+scale×action，ABAD scale.125，其余腿.25；轮速度5×action；处理后目标clip±100。M1力矩限幅腿180N·m、轮28.68N·m。未套用B2W质量、惯量、轮径或力矩上限；没有虚构已辨识的扭矩–转速曲线或速度硬限位。

完整配置使用16个原生torque motor，1000Hz显式PD+延迟+限幅，避免通用servo随机化把轮速度增益写成位置增益；延迟同时覆盖腿和轮。机体系根部速度按当前qvel与随机质心计算，避免原生IMU sensor旧一个物理步的问题。物理步保持1ms、策略50Hz、延迟0–20ms；完整配置使用implicit积分器，改善耦合多刚体高速旋转时的数值稳定性。仅从2ms缩到1ms未能消除此前的失稳。

MuJoCo只有一种sliding friction，不能与PhysX独立static [.3,1] / dynamic [.3,.8]完全等价。本实现μ∈[.3,1]、torsional=.005、rolling=.0001。恢复系数e近似映射到solref=[.02/zeta,zeta]，zeta=−ln(e)/sqrt(π²+ln(e)²)，e=0时zeta=1；timeconst×zeta恒为.02，避免恢复系数随机化同时放大接触刚度。实际反弹还依赖几何、时步与求解器。机器人geom priority1确保随机材料控制地面接触；self collision关闭，与B2W一致。

“完整迁移”指所有启用奖励与随机化类别都有实现；不声称两个物理引擎数值等价，也不将短验证当作收敛或实机能力证明。

## 参考与复验

固定robot_lab commit `09f6a9dfdf48f32f38bb851dfa3a7d44db32b270`：[B2W rough](https://github.com/fan-ziqi/robot_lab/blob/09f6a9dfdf48f32f38bb851dfa3a7d44db32b270/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_b2w/rough_env_cfg.py)、[父配置](https://github.com/fan-ziqi/robot_lab/blob/09f6a9dfdf48f32f38bb851dfa3a7d44db32b270/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)、[reward公式](https://github.com/fan-ziqi/robot_lab/blob/09f6a9dfdf48f32f38bb851dfa3a7d44db32b270/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py)、[PPO](https://github.com/fan-ziqi/robot_lab/blob/09f6a9dfdf48f32f38bb851dfa3a7d44db32b270/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_b2w/agents/rsl_rl_ppo_cfg.py)。

```bash
uv run --no-project --python .venv/bin/python python -m pytest tests -q
uv run --no-project --python .venv/bin/python python -m m1_unilab.validate_full --num-envs 512 --steps 1100
```

本次工作区可将Python路径改为 `../../work/unilab-venv/bin/python`。修复后验证结果见 `stability_fix_validation.json` 和 `validation_full_mixed_512.json`。

## 修复后验收

原64轮短测没有覆盖第170轮附近出现的故障，旧验收记录及26.35小时估算仅作历史记录，不能用于判断修复后配置。

修复新增了保存故障状态的真实MuJoCo回放回归用例、材质刚度不变量检查，以及地形净空、楼梯边缘与部分环境重置检查。512环境原生验证还独立检查随机出生无穿透、实际质量/惯量/质心/材质提交、逐物理步控制与持续外力、reset生命周期。

最终测试数量、512环境300次PPO更新、原checkpoint恢复验证和耗时以[修复后验收摘要](stability_fix_validation.json)为准。[定位与修复说明](stability_fix.md)、[512环境原生检查](validation_full_mixed_512.json)、[正式30k解析配置](full_training_config.json)。有限验证不构成全训练时长稳定性或策略收敛保证。

当前碰撞器已改为libccd，绕开2026-11-11第1046轮的原生CCD段错误。详见 [碰撞器修复与续训](native_collision_fix.md)。旧300轮验收属于切换前配置。
