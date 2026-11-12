# M1 四轮足 locomotion 框架调研

核对日期：2026-11-10。本文以一手仓库、原作者说明和厂商资料为依据；“建议”是工程判断，不表示已经完成 M1 训练或实机验证。

## 结论

首选 **robot_lab v2.3.2 + Isaac Lab v2.3.2 + Isaac Sim 5.1**，以 B2W/Go2W 的轮足任务为接口参考，单独增加 M1 资产和任务。理由是已有 12 个腿位置动作与 4 个轮速度动作、粗糙地形基础环境、PPO、随机化和配置注册机制；相比从纯四足框架增加轮子，迁移面更小。锁定 `robot_lab` commit `09f6a9dfdf48f32f38bb851dfa3a7d44db32b270`。其主线已迁往 Isaac Lab 3.0，不能混用主线安装说明和 2.3.2 配置。[版本依赖表](https://github.com/fan-ziqi/robot_lab/blob/main/README.md)、[固定版本](https://github.com/fan-ziqi/robot_lab/tree/09f6a9dfdf48f32f38bb851dfa3a7d44db32b270)

这属于具有完整工程结构的研究训练起点，不是可直接迁移到 M1 的成熟产品策略。可复用任务、算法和训练工具；不能把 Go2W/B2W checkpoint、执行器参数或电机坐标直接用于 M1。

## 候选比较

| 候选 | 核对到的事实 | 对本任务的判断 |
|---|---|---|
| robot_lab v2.3.2 | Go2W/B2W 两种轮足资产与 flat/rough 任务；Apache-2.0；外部 Isaac Lab extension | 首选。结构清楚，可减少自研环境代码；需覆盖上游自起身配置 |
| zhaozijie2022/LocoWheeledLegged | Isaac Lab 的 Go2W 项目，展示草地、楼梯、石块等实机演示；有动作滤波、历史/编码器策略与部署模块 | 备选及算法参考。当前 clone 未见顶层 LICENSE；README 同时出现 Sim 5.5.1 字样与 5.1.0 安装命令，版本需自行固定，不能只照 README 复制 |
| ShengqianChen/DreamWaQ_Go2W | Isaac Gym Preview 4，16 动作、5 帧观测历史、特权 critic、地形和速度课程，含 Go2W 部署 | 有价值的盲行走/地形奖励参考；新机器优先使用 Isaac Lab 路线，避免先背负旧运行时适配 |
| ShengqianChen/go2w_rl_gym | Go2W 训练与物理部署的轻量项目；BSD-3-Clause | 简单基线备选；总体仍是旧 Isaac Gym 技术路线 |
| unitreerobotics/unitree_rl_lab | 官方项目，Apache-2.0；当前 README 训练支持 Go2、H1、G1；树中虽有 Go2W deploy 配置，但没有相应轮足训练任务 | 可参考部署导出和官方模型来源；“有 Go2W deploy 目录”不等于已提供完整轮足 rough training |
| clearlab-sustech/Wheel-Legged-Gym | 实际配置是左右两腿、两轮，共 6 个关节，有 VMC 分支 | 名称容易误导；不是 M1 所需的四轮足直接基座 |

来源：[robot_lab B2W 配置](https://github.com/fan-ziqi/robot_lab/blob/09f6a9dfdf48f32f38bb851dfa3a7d44db32b270/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_b2w/rough_env_cfg.py)、[LocoWheeledLegged README](https://github.com/zhaozijie2022/LocoWheeledLegged/blob/015b8230b1a21331ab6d2b054a2ef8f1e435db84/README.md)、[DreamWaQ Go2W 参数](https://github.com/ShengqianChen/DreamWaQ_Go2W/blob/b64f7b90856fc71595870cb283ce29dd0f4d6f0f/legged_gym/legged_gym/envs/go2w/go2w_config.py)、[go2w_rl_gym](https://github.com/ShengqianChen/go2w_rl_gym/tree/df5dfcde9b62eaff5e422cfcaff77952846ece08)、[Unitree RL Lab README](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/README.md)、[Wheel-Legged-Gym 关节配置](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py)。

## 电机 kp/kd：A2W 优先，A2 腿部作为已授权备选

用户最终限定：**只参考各电机 kp/kd；质量、惯性、几何仍使用 M1；A2W 找不到可用 A2；轮子按最新要求参考 B2W 的 kp=0、kd=1。** 腿部保留已确认的 A2 增益，不采用整机质量替代方案。

### 可落地的 A2 腿部参数

找到 Unitree 官方 `unitree_rl_mjlab` 的 A2 训练和部署配置，两处数值一致。固定 commit：`1425b15f73bd4095f0df53709d7c389c3eb9e790`。

| 关节类别（四条腿同值） | kp / stiffness | kd / damping | 源控制方式 |
|---|---:|---:|---|
| hip：髋外展/内收 | 100 | 4 | BuiltinPositionActuatorCfg，位置 PD |
| thigh：大腿俯仰 | 100 | 4 | BuiltinPositionActuatorCfg，位置 PD |
| calf：膝关节 | 150 | 6 | BuiltinPositionActuatorCfg，位置 PD |
| 四个驱动轮 | 0（B2W 参考） | 1（B2W 参考） | 纯轮速目标；A2 纯腿源没有轮电机 |

来源一：[A2 命名执行器配置](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/assets/robots/unitree_a2/a2_constants.py)。代码按 `.*hip_.*`、`.*thigh_.*`、`.*calf_.*` 分组三种执行器，分别设置上述 stiffness/damping。仅借这两个字段；该文件中的 effort_limit、armature、默认姿态等不在本次替代范围。

来源二：[A2 部署参数](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/deploy/robots/a2/config/policy/velocity/v0/params/deploy.yaml)。文件的 stiffness 为 `[100,100,150]` 重复四次，damping 为 `[4,4,6]` 重复四次；策略步长 0.02 s，动作类型 `JointPositionAction`。它还定义自己的 `joint_ids_map`，迁移到 M1 必须按 M1 关节语义映射，不能复用该索引表。[A2 部署状态代码](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/deploy/robots/a2/src/State_RLBase.cpp) 加载该 deploy.yaml 并将处理后的动作写入各 motor 的 q 指令。

对关节侧力矩 PD 式 `tau = kp * (q_target - q) + kd * (dq_target - dq) + tau_ff`，若 q 用 rad、dq 用 rad/s、tau 用 N·m，则 kp 单位为 N·m/rad，kd 为 N·m·s/rad。这是按该位置执行器物理含义作的量纲说明；源配置没有另行打印单位。上述数值是**官方 A2 策略的控制增益**，不是 A2W 参数，也不是电机系统辨识结果。借给 M1 后仍须通过 M1 仿真响应与实际控制定义验证。

### A2W 专项检索结果与边界

截至本次核对，未找到可验证的 A2W 全 16 电机或 hip/thigh/calf/wheel 分组 kp/kd 表。已排查：

- 官方 `unitree_sdk2` commit `aaf69cb56c9d5ed63d8b284008ec3a8ab999bec1`：A2 示例目录含 sport/audio，没有 A2W 专用 lowlevel 参数文件。[A2 示例目录](https://github.com/unitreerobotics/unitree_sdk2/tree/aaf69cb56c9d5ed63d8b284008ec3a8ab999bec1/example/a2)
- 官方 `unitree_sdk2_python` commit `814556d15970dd2ecf1c9984e845ca02ab07e206`：A2 同样是 sport/audio；没有 A2W 专用 kp/kd 配置。[A2 Python 示例](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206/example/a2)
- 官方 `unitree_rl_mjlab` commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`：存在上面给出的 A2 纯腿 12 电机配置，没有 A2W 任务/四轮增益。[官方仓库](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/1425b15f73bd4095f0df53709d7c389c3eb9e790)
- 官方 `unitree_ros2` commit `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88`、`unitree_rl_lab` commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` 目录中未见 A2W 专用任务或配置。[ROS2](https://github.com/unitreerobotics/unitree_ros2/tree/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88)、[RL Lab](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3)
- A2W 开发者项目 `Jojoisnyze/Masterarbeit-Unitree-A2W` 确有 A2W 标签及 `a2_wheel.urdf`，但读到的 `REG.Kp=3.0` 是 Z1 机械臂相机姿态外环增益 `[1/s]`，并非底盘电机 PD；URDF 中未见 kp/kd。[相机稳定控制器](https://github.com/Jojoisnyze/Masterarbeit-Unitree-A2W/blob/dc5d18a622d5ca3d4accb07fccc93ac0b54e5b48/Matlab/parameter/par_regler.m)、[A2W URDF](https://github.com/Jojoisnyze/Masterarbeit-Unitree-A2W/blob/dc5d18a622d5ca3d4accb07fccc93ac0b54e5b48/X2-0807urdf/a2_wheel.urdf)
- `Autodiscovery/a2w-simpleslam` 当前为空仓库，不能提供控制参数。[仓库](https://github.com/Autodiscovery/a2w-simpleslam)
- [Unitree A2 开发文档入口](https://support.unitree.com/home/en/A2_SDK_Development_Guide) 能定位，但网页抓取返回空正文，直接请求返回 HTTP 567，当前无可用浏览器完成读取。因此不能声称已验证其全部正文不存在 A2W 参数。

检索使用 A2W、A2-W、a2_w 等型号变体并组合 kp/kd、stiffness/damping、LowCmd、lowlevel、底层控制等关键词，同时核对官方仓库目录。此结论限于本次可访问公开来源；不排除厂商开发包中存在更多参数。当前实施采用已授权的 **A2 腿部 kp/kd 备选**，四轮按用户最新要求改为 B2W 参考的 **0/1**，动作缩放参考 B2W 取 5 rad/s。该轮参数来自社区训练配置，不称为宇树官方 sysid。[B2W 轮执行器来源](https://github.com/fan-ziqi/robot_lab/blob/09f6a9dfdf48f32f38bb851dfa3a7d44db32b270/source/robot_lab/robot_lab/assets/unitree.py)

## robot_lab 迁移时必须覆盖的细节

上游 B2W actor 移除了基座线速度和高度扫描，critic 保留仿真特权信息；默认腿位置与轮速度分别控制。这适合作为不依赖在线地形感知的基础策略，但并不保证盲策略能够可靠通过任意台阶。上游 reset 将 roll/pitch 随机到 ±π，并关闭 `illegal_contact` 终止，是兼顾恢复/自起身的设置。M1 基础 locomotion 建议改为站姿附近初始化，躯干接触/倾倒终止，先训练通行再独立研究自起身。[B2W rough 配置](https://github.com/fan-ziqi/robot_lab/blob/09f6a9dfdf48f32f38bb851dfa3a7d44db32b270/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_b2w/rough_env_cfg.py)

另外，`joint_pos_rel_without_wheel` 先按 `asset_cfg.joint_ids` 重排/切片，再用全局 `wheel_asset_cfg.joint_ids` 清零。因此在 M1 自定义排序时存在位置索引不一致风险。直接让位置观测只选 12 个腿关节更清楚；速度观测选 16 个驱动关节，动作按固定名称映射，不能依赖 URDF/PhysX 自然排序。[观测实现](https://github.com/fan-ziqi/robot_lab/blob/09f6a9dfdf48f32f38bb851dfa3a7d44db32b270/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/observations.py)

基础 actor 若采用角速度 3、重力投影 3、速度命令 3、腿相对位置 12、全关节速度 16、上一动作 16，单帧共 **53 维**；这是建议接口计算，不是上游原配置维度。若后续加入历史、状态估计器或 terrain perception，需一并改训练、导出和推理的数据契约。

不要给轮子沿用普通点足的水平滑动惩罚：正常滚动时轮轴中心有水平速度；要区分允许的滚动与侧滑，否则会把策略推向锁轮或拖轮。平地保留低动作变化/能耗、速度跟踪；爬台阶不要用过强的静态姿态惩罚阻止抬腿。该段为对轮足机构的工程推论。

## 地形与验证建议

建议将目标解释为“在明确尺寸范围内覆盖四类地形”，不是“各种地形均可通过”。训练先以平地和低扰动建立稳定驱动，再混入坡度、粗糙表面与上/下楼梯，逐渐增加难度；同时保留低难度样本。每类分别评估随机种子、速度/航向命令、质量/摩擦/延迟变化下的成功率、速度误差、摔倒率和峰值扭矩。训练 reward 升高不等于能过楼梯。

M1 未验证的 SDK 坐标、关节零位、方向、动作顺序，应在离线映射与仿真检查中独立验收。仅换 URDF 不足以称完成 sim2real。

## 环境与复现边界

上述 2.3.2/5.1 组合是为了固定已有任务 API。NVIDIA 当前的 Isaac Sim 5.1 文档已标为不再支持，未来维护应另建升级分支。其当前 requirements 页面列 Ubuntu 22.04/24.04、Linux 测试驱动 580.65.06，并将 RTX 5080/Blackwell 列入硬件等级；训练还需额外内存与显存。先在目标训练机做资产导入、短 rollout、短 PPO 更新，再启动长训练，不能凭配置静态通过就声称 GPU 训练可用。[Isaac Sim 5.1 要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

本次研究仅核对源码与公开资料，未执行 Isaac Sim 训练，没有产出训练 checkpoint、地形成功率或实机验证结果。
