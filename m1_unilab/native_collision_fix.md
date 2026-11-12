# 原生碰撞器崩溃处理与第1000号权重续训

2026-11-11 00:44:33（新加坡时间），PID64722在MuJoCo3.11.0的
`projectOriginPlane → mjc_ccd → mjc_penetration → mjc_ConvexHField`
调用链发生SIGSEGV。最后记录迭代1046；第1000号checkpoint可正常读取。

完整配置的solver片段现设置 `nativeccd="disable"`，让凸几何及高度场
碰撞走libccd的MPR实现，避开本次故障调用路径。仍保留全部地形、接触、
奖励、随机化、implicit积分与出生净空处理；M1资产和电机基准不变。
两种碰撞算法的接触点和穿透距离可能不同，不能声称轨迹或收敛完全等价。

[MuJoCo3.11官方分支代码](https://github.com/google-deepmind/mujoco/blob/3.11.0/src/engine/engine_collision_convex.c#L80-L85)
确认该选项切换到libccd。[上游问题3646](https://github.com/google-deepmind/mujoco/issues/3646)
记录了类似原生EPA越界崩溃，但此次没有保存故障瞬间物理状态，尚未证明
与该问题是同一个越界缺陷。本次属于绕开已确认故障路径，不是上游C代码补丁。

验证包括真实编译模型的碰撞器标志断言（修改前失败、修改后通过）、
全部48项测试、512环境100策略步物理检查，以及从第1000号权重进行12次PPO
更新的独立续训验证。原穿透测试改为检查“深穿透经净空修正后消失”，
不要求libccd返回与旧EPA完全相同的穿透距离；净空标准没有降低。

恢复权重保留为 `checkpoints/recovery_1000.pt`，校验值和来源见同名JSON。
累计12312576条转移，除以512×24是1002次更新。正式续训追加28998次，
合计30000次；12次验证的结果不并入正式训练。从保存状态恢复网络与优化器，
仿真状态重新初始化，地形课程也重新从初始级别运行。

`resume_mac_full.sh` 已改用该权重。`background_train.py` 负责独立进程启动、
完整stdout/stderr日志以及退出状态记录；`PYTHONFAULTHANDLER=1`开启底层故障
诊断。后台运行记录位于 `training_records/full_from_1000/`。
关闭终端不会使这个已分离的训练进程退出；电脑关机、重启或强制睡眠仍会影响运行。
有限验证不能保证完整3万轮运行或策略收敛。
