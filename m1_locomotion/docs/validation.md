# 已执行验证与待运行项目

验证环境：当前 macOS ARM64，隔离 Python 3.12.13，MuJoCo 4.0.0、NumPy 2.5.4、pytest 9.2.0。此环境用于 CPU 检查，Linux 训练环境另按 README 安装。

## 已执行

- 11 项 pytest 检查通过：运动关节集合与树结构、有效惯量、质量与惯量逐项保留、站姿限位、四轮轴心高度一致、轮旋转不改变轴心位置、动作维度/顺序/缩放/限幅、非法输入拒绝、资产生成可复现、A2 关节功能映射、用户轮增益缺失时不会回退到测试值。已删除不再使用的位置斜坡实现及其 3 项测试。
- Python 全部源码语法编译、安装脚本 bash 语法检查通过。
- MuJoCo 从处理后 URDF 导入，33 个 CAD link 中固定关节被引擎合并，保留 16 个驱动自由度与浮动基座。
- 10 秒名义 PD 物理检查：前 5 秒站立，后 5 秒四轮目标 5 rad/s。使用已核对的 A2 腿部增益与用户授权参考 B2W 的正式轮 kp=0、kd=1，轮子采用纯速度反馈。

本次 CPU 结果：质量 42.338 kg，最低机身高度约 0.48590 m，最大倾角约 0.02246 rad；滚动阶段前进约 2.37424 m，最大腿力矩约 25.329 N·m、轮力矩约 5.00147 N·m，未发生非轮部位触地。完整数值见 `cpu_smoke_results.json`。这只证明所测条件下的模型与控制契约基本一致，不是训练出的策略。

本次命令（从工程上层工作区运行）：`UV_CACHE_DIR="$PWD/work/uv-cache" uv run --no-project --python work/venv/bin/python -m pytest outputs/m1_locomotion/tests -q`；`PYTHONPATH="$PWD/outputs/m1_locomotion" UV_CACHE_DIR="$PWD/work/uv-cache" uv run --no-project --python work/venv/bin/python outputs/m1_locomotion/scripts/smoke_mujoco.py --output outputs/m1_locomotion/docs/cpu_smoke_results.json`。

CPU 检查采用零延迟、无随机化和 MuJoCo 接触模型；未验证 Isaac PhysX 的接触、延迟执行器或 GPU 环境配置。

## 尚未执行

本页记录 Isaac Lab 扩展和 CPU 模型检查的状态。当前尚未在 Isaac Sim 中运行 M1 资产导入、8 环境短 rollout、PPO 梯度更新、平地预训练、混合地形训练或 ONNX 导出。该后端未生成 checkpoint，也没有斜坡、粗糙地面或楼梯通过率。Mac UniLab 的迁移与训练结果应按其独立运行记录评估。

先运行 `smoke_isaac.py`，验证真实 action joint order、53/243 维观测和有限物理状态，再做 64 环境/5 次更新的短训练。通过之后才按 README 启动长训练。`evaluate.py` 必须读取真实 checkpoint 才能生成指标；现有文件中没有预填成功率。

## 建议验收口径

每类地形至少 3 个随机种子、每次 100 环境，分别记录难度 0/0.5/1 的结果。先对名义参数评估，再开启随机化。基础阶段可以将“平地通过率 ≥95%，指定范围内各非平地类别 ≥90%”作为待验证的工程目标；这些百分比是建议验收门槛，并非已有结果。还应核对峰值力矩、速度误差、抬腿行为和视频，不能仅按平均 reward 接受策略。
