# M1_loco

M1 轮足机器人的 locomotion 训练（UniLab + MuJoCo，Apple Silicon）。

- `m1_locomotion/`：M1 模型、资产与基础任务包
- `m1_unilab/`：训练、评测、录像、键盘控制预览；详细说明见 `m1_unilab/README.md`
- `m1_unilab/checkpoints/full_27000.pt`：b2w-full 混合地形训练第 27000 轮权重（训练未跑完 30000 轮）
- `m1_unilab/evaluations/`：第 10000、24000 轮的固定条件评测（vx=0.5 m/s，每条件 8 环境，单 seed，仅名义仿真，不代表实机可用）

两个目录需并排放置，先执行 `bash m1_unilab/setup_mac.sh`。
键盘控制：`bash m1_unilab/control_mac.command --checkpoint m1_unilab/checkpoints/full_27000.pt`，浏览器打开 http://127.0.0.1:8765，W/S 前后、A/D 转向、Q/E 平移、Shift 加速、空格归零。

已知短板：下坡、下楼梯、较高台阶上行评测未通过。
