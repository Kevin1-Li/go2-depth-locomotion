import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Teacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:TeacherRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.teacher_env_cfg:TeacherRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.teacher_rsl_rl_ppo_cfg:TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-BlindEval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blind_eval_env_cfg:BlindEvalEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.blind_eval_env_cfg:BlindEvalPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.blind_eval_rsl_rl_ppo_cfg:BlindEvalPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Depth-Student",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_env_cfg:StudentRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.student_env_cfg:StudentRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.student_distill_rsl_rl_cfg:DistillationRunnerCfg",
    },
)
