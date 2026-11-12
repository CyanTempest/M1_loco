# High-contact reset fixture

`high_contact_reset.npz` records a real native M1 failure with the original full
profile at 1ms: frozen user checkpoint100, reset batch11, environment312.
It contains initial reset state, committed mass/inertia/COM/material arrays,
97 actual substep torque commands and world-frame wrenches, and the failed
published trajectory. The recorded final trajectory intentionally contains Inf.

Independent MuJoCo replay matches the recorded float32 qpos/qvel exactly and
reproduces BADQACC at .093s followed by BADQPOS/BADQVEL at .095s. Tests generate
the current M1 terrain/model from production configuration, then replay these
unchanged inputs. No third-party robot physics model or synthetic NaN injection
is used by this fixture.
