from gym.envs.registration import register
from SafetyGuided_DRL.envs.monitor import Monitor
from SafetyGuided_DRL.envs.monitor import SafeMonitor

# Classic
# ----------------------------------------

register(
    id='CartPoleGP-v0',
    entry_point='SafetyGuided_DRL.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPoleGP-v1',
    entry_point='SafetyGuided_DRL.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='MountainCarGP-v0',
    entry_point='SafetyGuided_DRL.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='MountainCarContinuousGP-v0',
    entry_point='SafetyGuided_DRL.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='PendulumGP-v0',
    entry_point='SafetyGuided_DRL.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='AcrobotGP-v1',
    entry_point='SafetyGuided_DRL.envs.classic_control:AcrobotEnv',
    max_episode_steps=500,
)

# Box2d
# ----------------------------------------

register(
    id='LunarLanderGP-v2',
    entry_point='SafetyGuided_DRL.envs.box2d:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuousGP-v2',
    entry_point='SafetyGuided_DRL.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='BipedalWalkerGP-v2',
    entry_point='SafetyGuided_DRL.envs.box2d:BipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id='BipedalWalkerHardcoreGP-v2',
    entry_point='SafetyGuided_DRL.envs.box2d:BipedalWalkerHardcore',
    max_episode_steps=2000,
    reward_threshold=300,
)


# Mujoco
# ----------------------------------------

# 2D
register(
    id='ReacherGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Reacher3dGP-v1',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:Reacher3dEnv',
    max_episode_steps=500,
    reward_threshold=-200,
)

register(
    id='PusherGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='ThrowerGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='StrikerGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='InvertedPendulumGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulumGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id='HalfCheetahGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HopperGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='SwimmerGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Walker2dGP-v2',
    max_episode_steps=1000,
    entry_point='SafetyGuided_DRL.envs.mujocoGP:Walker2dEnv',
)

register(
    id='AntGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HumanoidGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidStandupGP-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoGP:HumanoidStandupEnv',
    max_episode_steps=1000,
)

# Baseline Environment

# Classic
# ----------------------------------------

register(
    id='CartPoleBL-v0',
    entry_point='SafetyGuided_DRL.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPoleBL-v1',
    entry_point='SafetyGuided_DRL.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='MountainCarBL-v0',
    entry_point='SafetyGuided_DRL.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='MountainCarContinuousBL-v0',
    entry_point='SafetyGuided_DRL.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='PendulumBL-v0',
    entry_point='SafetyGuided_DRL.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='AcrobotBL-v1',
    entry_point='SafetyGuided_DRL.envs.classic_control:AcrobotEnv',
    max_episode_steps=500,
)

# Box2d
# ----------------------------------------

register(
    id='LunarLanderBL-v2',
    entry_point='SafetyGuided_DRL.envs.box2d:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuousBL-v2',
    entry_point='SafetyGuided_DRL.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='BipedalWalkerBL-v2',
    entry_point='SafetyGuided_DRL.envs.box2d:BipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id='BipedalWalkerHardcoreBL-v2',
    entry_point='SafetyGuided_DRL.envs.box2d:BipedalWalkerHardcore',
    max_episode_steps=2000,
    reward_threshold=300,
)


# mujocoBL
# ----------------------------------------

# 2D
register(
    id='ReacherBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Reacher3dBL-v1',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:Reacher3dEnv',
    max_episode_steps=500,
    reward_threshold=-200,
)

register(
    id='PusherBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='ThrowerBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='StrikerBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='InvertedPendulumBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulumBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id='HalfCheetahBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HopperBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='SwimmerBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Walker2dBL-v2',
    max_episode_steps=1000,
    entry_point='SafetyGuided_DRL.envs.mujocoBL:Walker2dEnv',
)

register(
    id='AntBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HumanoidBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidStandupBL-v2',
    entry_point='SafetyGuided_DRL.envs.mujocoBL:HumanoidStandupEnv',
    max_episode_steps=1000,
)
