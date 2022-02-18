from gym.envs.registration import register

register(
    id='merging_env-v0',
    entry_point='merging_gym.envs:MergeEnv',
)

register(
    id='merging_env_extend-v0',
    entry_point='merging_gym.envs:MergeEnvExtend',
)
