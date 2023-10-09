# noqa: D104
from gymnasium.envs.registration import register

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.maze import maps
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v0


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    def _merge(a, b):
        a.update(b)
        return a

    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }

        # Fetch
        register(
            id=f"FetchSlide{suffix}-v1",
            entry_point="gymnasium_robotics.envs.fetch.slide:MujocoPyFetchSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchSlide{suffix}-v2",
            entry_point="gymnasium_robotics.envs.fetch.slide:MujocoFetchSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPickAndPlace{suffix}-v1",
            entry_point="gymnasium_robotics.envs.fetch.pick_and_place:MujocoPyFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPickAndPlace{suffix}-v2",
            entry_point="gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v1",
            entry_point="gymnasium_robotics.envs.fetch.reach:MujocoPyFetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v2",
            entry_point="gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPush{suffix}-v1",
            entry_point="gymnasium_robotics.envs.fetch.push:MujocoPyFetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPush{suffix}-v2",
            entry_point="gymnasium_robotics.envs.fetch.push:MujocoFetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )


        # Hand
        register(
            id=f"HandReach{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.reach:MujocoPyHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"HandReach{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.reach:MujocoHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"HandManipulateBlockRotateZ{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )
        register(
            id=f"PrivilegedHandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:PrivilegedMujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                    "camera_names": ["camera_hand"],
                    "log_image_keys": [True],
                    "render_mode": "rgb_array",
                    "touch_visualisation": "off",
                    "width": 64,
                    "height": 64,
                },
                kwargs,
            ),
            max_episode_steps=100,
        )
        register(
            id=f"PrivilegedHandManipulateBlockRotateZ_ImageBooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:PrivilegedMujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                    "camera_names": ["camera_hand"],
                    "log_image_keys": [False],
                    "render_mode": "rgb_array",
                    "touch_visualisation": "off",
                    "include_initial_object_state": False,
                    "width": 64,
                    "height": 64,
                },
                kwargs,
            ),
            max_episode_steps=100,
        )
        register(
            id=f"HORAPrivilegedHandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:HORAPrivilegedMujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                    "render_mode": "rgb_array",
                    "touch_visualisation": "off",
                    "width": 64,
                    "height": 64,
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )
        register(
            id=f"PrivilegedHandManipulateBlockRotateXYZ_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:PrivilegedMujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                    "camera_names": ["camera_hand"],
                    "log_image_keys": [True],
                    "render_mode": "rgb_array",
                    "touch_visualisation": "off",
                    "width": 64,
                    "height": 64,
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockFull{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockFull{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateBlock{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )
        register(
            id=f"PrivilegedHandManipulateBlock_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:PrivilegedMujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                    "camera_names": ["camera_hand"],
                    "log_image_keys": [True],
                    "render_mode": "rgb_array",
                    "touch_visualisation": "off",
                    "width": 64,
                    "height": 64,
                },
                kwargs,
            ),
            max_episode_steps=100,
        )


        register(
            id=f"HandManipulateBlock_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggFull{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggFull{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateEgg{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"PrivilegedHandManipulatePenRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:PrivilegedMujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                    "camera_names": ["camera_hand"],
                    "log_image_keys": [True],
                    "render_mode": "rgb_array",
                    "touch_visualisation": "off",
                    "width": 64,
                    "height": 64,
                },
                kwargs,
            ),
            max_episode_steps=100,
        )
        register(
            id=f"HORAPrivilegedHandManipulatePenRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:HORAPrivilegedMujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                    "camera_names": ["camera_hand"],
                    "log_image_keys": [True],
                    "render_mode": "rgb_array",
                    "touch_visualisation": "off",
                    "width": 64,
                    "height": 64,
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"TGRLHandManipulatePenRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:TGRLMujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                    "render_mode": "rgb_array",
                    "touch_visualisation": "off",
                    "include_initial_object_state": False,
                    "include_teacher_state": True,
                    "width": 64,
                    "height": 64,
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"PrivilegedHandManipulatePenRotate_ImageBooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:PrivilegedMujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                    "camera_names": ["camera_hand"],
                    "log_image_keys": [False],
                    "render_mode": "rgb_array",
                    "touch_visualisation": "off",
                    "include_initial_object_state": False,
                    "width": 64,
                    "height": 64,
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenFull{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenFull{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        #####################
        # D4RL Environments #
        #####################

        # ----- AntMaze -----

        register(
            id=f"AntMaze_UMaze{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )
        register(
            id=f"AntMaze_Open_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Medium{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        # ----- Marble Env -----
        for observation_type in ["PO", "FO"]:
            for pixel_ob in ["pixels", "camera", "state", "depth"]:
                register(
                    id=f"SweepMarbles{observation_type}{pixel_ob.capitalize()}{reward_type.capitalize()}-v0",
                    entry_point="gymnasium_robotics.envs.marbles.marbles:SweepMarblesEnv",
                    disable_env_checker=True,
                    kwargs={
                        "observation_type": observation_type,
                        "pixel_ob": pixel_ob,
                        "reward_type": reward_type,
                        "render_mode": "rgb_array",
                        "width": 64,
                        "height": 64,
                    },
                    max_episode_steps=100,
                )

        # ----- PointMaze -----
        def add_cameras(tree, maze_size_scaling):
            import xml.etree.ElementTree as ET
            # Add cameras.
            cam_body = tree.find(".//worldbody/body")
            worldbody = tree.find(".//worldbody")
            assert cam_body is not None and cam_body.get('name') in {'particle'}

            M = maze_size_scaling
            ET.SubElement(
                cam_body, 
                "camera",
                name="ego_cam",
                mode="track",
                pos="0 0 1.8",
            )
            quadrant_cam_config = [
                {'pos': f'0 {-1.25*M} {4.25*M}', 'mode': 'fixed', 'euler': '0 0 0', 'fovy': '60'},
                {'pos': f'{1.25*M} 0 {5.25*M}', 'mode': 'fixed', 'euler': '0 0 0', 'fovy': '60'},
                {'pos': f'0 {1.25 * M} {2.25*M}', 'mode': 'fixed', 'euler': '0 0 0', 'fovy': '60'} # top hallway,
            ]        
            for i, config in enumerate(quadrant_cam_config):
                cam_body = ET.SubElement(
                    worldbody,
                    "body",
                    name=f"q{i}_cam_body",
                )
                ET.SubElement(
                    cam_body, 
                    "site",
                    name=f"q{i}_cam_site",
                    rgba="0 1.0 0 1.0",
                    size="0.05",
                    pos=config['pos'],
                )
                ET.SubElement(
                    cam_body, 
                    "camera",
                    name=f"q{i}_cam",
                    **config,
                )
            # Add curtains.
            curtain_config = [
                {'pos': f'{0.5*M} 0 0', 'rgba': '0 0 0 1', 'type': 'box', 'size': f'0.05 {10/4 * M} 10'},
                {'pos': f'{-0.5*M} {0.5*M} 0', 'rgba': '0 0 0 1', 'type': 'box', 'size': f'{4/4 * M} 0.05 10'}
            ]
            for i, config in enumerate(curtain_config):
                ET.SubElement(
                    worldbody, 
                    "site",
                    name=f"curtain_{i}",
                    **config
                )

        register(
            id=f"VisualPointMaze_UMaze{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:VisualPointMazeEnv",
            disable_env_checker=True,
            kwargs={
                "maze_map": maps.VISUAL_U_MAZE,
                "maze_size_scaling": 1.0,
                "render_mode": "rgb_array",
                "camera_names": ["ego_cam", "q0_cam", "q1_cam", "q2_cam"],
                "width": 32,
                "height": 32,
                "maze_fns": [add_cameras],
                "continuing_task": False,
                "position_noise_range": 0.001,
            },
            max_episode_steps=500,
        )
        register(
            id=f"VisualPointMaze_UMaze{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.point_maze:VisualPointMazeEnv",
            disable_env_checker=True,
            kwargs={
                "maze_map": maps.VISUAL_U_MAZE,
                "maze_size_scaling": 1.5,
                "render_mode": "rgb_array",
                "camera_names": ["ego_cam", "q0_cam", "q1_cam", "q2_cam"],
                "width": 32,
                "height": 32,
                "maze_fns": [add_cameras],
                "continuing_task": False,
                "position_noise_range": 0.001,
            },
            max_episode_steps=500,
        )

        register(
            id=f"VisualPointMaze_UMaze{suffix}-v5",
            entry_point="gymnasium_robotics.envs.maze.point_maze:VisualPointMazeEnv",
            disable_env_checker=True,
            kwargs={
                "maze_map": maps.VISUAL_U_MAZE,
                "maze_size_scaling": 1.25,
                "render_mode": "rgb_array",
                "camera_names": ["ego_cam", "q0_cam", "q1_cam", "q2_cam"],
                "width": 32,
                "height": 32,
                "maze_fns": [add_cameras],
                "continuing_task": False,
                "position_noise_range": 0.001,
            },
            max_episode_steps=500,
        )

        register(
            id=f"VisualPointMaze_UMaze{suffix}-v6",
            entry_point="gymnasium_robotics.envs.maze.point_maze:VisualPointMazeEnv",
            disable_env_checker=True,
            kwargs={
                "maze_map": maps.VISUAL_U_MAZE,
                "maze_size_scaling": 1.0,
                "render_mode": "rgb_array",
                "camera_names": ["ego_cam", "q0_cam", "q1_cam", "q2_cam"],
                "width": 32,
                "height": 32,
                "maze_fns": [add_cameras],
                "continuing_task": False,
                "position_noise_range": 0.001,
            },
            max_episode_steps=250,
        )

        register(
            id=f"VisualPointMaze_UMaze{suffix}-v7",
            entry_point="gymnasium_robotics.envs.maze.point_maze:VisualPointMazeEnv",
            disable_env_checker=True,
            kwargs={
                "maze_map": maps.VISUAL_U_MAZE,
                "maze_size_scaling": 4.0,
                "render_mode": "rgb_array",
                "camera_names": ["ego_cam", "q0_cam", "q1_cam", "q2_cam"],
                "width": 32,
                "height": 32,
                "maze_fns": [add_cameras],
                "continuing_task": False,
                "position_noise_range": 0.001,
            },
            max_episode_steps=500,
        )

        register(
            id=f"PointMaze_UMaze{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Medium{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Medium_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Medium_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Large{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

        register(
            id=f"PointMaze_Large_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

        register(
            id=f"PointMaze_Large_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

    for reward_type in ["sparse", "dense"]:
        suffix = "Sparse" if reward_type == "sparse" else ""
        version = "v1"
        kwargs = {
            "reward_type": reward_type,
        }

        register(
            id=f"AdroitHandDoor{suffix}-{version}",
            entry_point="gymnasium_robotics.envs.adroit_hand.adroit_door:AdroitHandDoorEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

        register(
            id=f"AdroitHandHammer{suffix}-{version}",
            entry_point="gymnasium_robotics.envs.adroit_hand.adroit_hammer:AdroitHandHammerEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

        register(
            id=f"AdroitHandPen{suffix}-{version}",
            entry_point="gymnasium_robotics.envs.adroit_hand.adroit_pen:AdroitHandPenEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

        register(
            id=f"AdroitHandRelocate{suffix}-{version}",
            entry_point="gymnasium_robotics.envs.adroit_hand.adroit_relocate:AdroitHandRelocateEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

    register(
        id="FrankaKitchen-v1",
        entry_point="gymnasium_robotics.envs.franka_kitchen:KitchenEnv",
        max_episode_steps=280,
    )

    register(
        id=f"ObjectPushQuadHard-v0",
        entry_point="gymnasium_robotics.envs.fetch.push_quad:MujocoFetchPushQuadHardEnv",
        kwargs={
            "camera_names": ["camera_q1", "camera_q2", "camera_q3", "camera_q4","camera_overhead", "gripper_camera_rgb", "camera_under"],
            "reward_type": "dense",
            "action_space_type": "object",
            "render_mode": "rgb_array",
            "width": 32,
            "height": 32,
        },
        max_episode_steps=50,
    )

    register(
        id=f"FetchPushQuadPose-v0",
        entry_point="gymnasium_robotics.envs.fetch.push_quad:MujocoFetchPushQuadPoseEnv",
        kwargs={
            "camera_names": ["camera_q1"],
            "reward_type": "dense",
            "action_space_type": "object",
            "include_obj_pose": True,
            "render_mode": "rgb_array",
            "width": 64,
            "height": 64,
        },
        max_episode_steps=50,
        disable_env_checker=True,
    )
    register(
        id=f"FetchPushQuadPOPose-v0",
        entry_point="gymnasium_robotics.envs.fetch.push_quad:MujocoFetchPushQuadPoseEnv",
        kwargs={
            "camera_names": ["camera_q1"],
            "reward_type": "dense",
            "action_space_type": "object",
            "include_obj_pose": False,
            "render_mode": "rgb_array",
            "width": 64,
            "height": 64,
        },
        max_episode_steps=50,
        disable_env_checker=True,
    )

    register(
        id=f"ObjectPushQuadSparseHard-v0",
        entry_point="gymnasium_robotics.envs.fetch.push_quad:MujocoFetchPushQuadHardEnv",
        kwargs={
            "camera_names": ["camera_q1", "camera_q2", "camera_q3", "camera_q4","camera_overhead", "gripper_camera_rgb", "camera_under"],
            "reward_type": "sparse",
            "action_space_type": "object",
            "render_mode": "rgb_array",
            "width": 32,
            "height": 32,
        },
        max_episode_steps=50,
    )

    register(
        id=f"FetchPushQuadHard-v0",
        entry_point="gymnasium_robotics.envs.fetch.push_quad:MujocoFetchPushQuadHardEnv",
        kwargs={
            "camera_names": ["camera_q1", "camera_q2", "camera_q3", "camera_q4","camera_overhead", "gripper_camera_rgb", "camera_under"],
            "reward_type": "dense",
            "action_space_type": "robot",
            "render_mode": "rgb_array",
            "width": 32,
            "height": 32,
        },
        max_episode_steps=50,
    )
    register(
        id=f"FetchPushQuadSparseHard-v0",
        entry_point="gymnasium_robotics.envs.fetch.push_quad:MujocoFetchPushQuadHardEnv",
        kwargs={
            "camera_names": ["camera_q1", "camera_q2", "camera_q3", "camera_q4","camera_overhead", "gripper_camera_rgb", "camera_under"],
            "reward_type": "sparse",
            "action_space_type": "robot",
            "render_mode": "rgb_array",
            "width": 32,
            "height": 32,
        },
        max_episode_steps=50,
    )

    register(
        id=f"FetchPushQuadHardPO-v0",
        entry_point="gymnasium_robotics.envs.fetch.push_quad:MujocoFetchPushQuadHardEnv",
        kwargs={
            "camera_names": ["camera_overhead", "gripper_camera_rgb"],
            "reward_type": "dense",
            "action_space_type": "robot",
            "render_mode": "rgb_array",
            "width": 32,
            "height": 32,
        },
        max_episode_steps=50,
    )

    # ----- PointMaze from DMC -----
    register(
        id=f"DMCPointMaze-v0",
        entry_point="gymnasium_robotics.envs.point_maze.maze:PointEnv",
        disable_env_checker=True,
        kwargs={
            "render_mode": "rgb_array",
            "width": 32,
            "height": 32,
        },
        max_episode_steps=1000,
    )

    # ------ IndicatorBoxBlock from Robosuite ------
    for observation_mode in ["FO", "PO"]:
        register(
            id=f"{observation_mode}IndicatorBoxBlock-v0",
            entry_point="gymnasium_robotics.envs.occluded_manipulation.indicator_box_block:GymIndicatorBoxBlock",
            max_episode_steps=100,
            kwargs={
                "views": ["agentview", "sideview"] if observation_mode == "FO" else ["agentview"],
                "width": 64,
                "height": 64,
            },
        )

    # ------ OccludedPick ------
    for observation_mode in ["FO", "PO", "DepthFO", "DepthPO"]:
        register(
            id=f"{observation_mode}OccludedPick-v0",
            entry_point="gymnasium_robotics.envs.fetch.occluded_pick:FetchOccludedPickEnv",
            max_episode_steps=100,
            disable_env_checker=True,
            kwargs={
                "camera_names": ["external_camera_0", "behind_camera"] if "FO" in observation_mode else ["external_camera_0"],
                "width": 64,
                "height": 64,
                "render_mode": "depth_array" if "depth" in observation_mode.lower() else "rgb_array",
            },
        )

    # ------ Gripper Camera -> 2D Blind Pick ------
    for observation_mode in ["FO", "PO", "DepthFO", "DepthPO"]:
        for difficulty in [0.07, 0.15]:
            register(
                id=f"{observation_mode}Gripper2DBlind{int(difficulty*100)}cmPick-v0",
                entry_point="gymnasium_robotics.envs.fetch.blind_pick:FetchBlindPickEnv",
                max_episode_steps=100,
                disable_env_checker=True,
                kwargs={
                    "camera_names": ["gripper_camera_rgb"] if "FO" in observation_mode else None,
                    "width": 64,
                    "height": 64,
                    "render_mode": "depth_array" if "depth" in observation_mode.lower() else "rgb_array",
                    "obj_range": difficulty,
                },
            )

    # ------ Under/Gripper Camera -> 2D Blind Pick ------
    for observation_mode in ["FO", "PO"]:
        for difficulty in [0.07, 0.15]:
            register(
                id=f"{observation_mode}UnderGripper2DBlind{int(difficulty*100)}cmPick-v0",
                entry_point="gymnasium_robotics.envs.fetch.blind_pick:FetchBlindPickEnv",
                max_episode_steps=100,
                disable_env_checker=True,
                kwargs={
                    "camera_names": ["camera_under", "gripper_camera_rgb"] if "FO" in observation_mode else None,
                    "width": 64,
                    "height": 64,
                    "render_mode": "rgb_array",
                    "obj_range": difficulty,
                },
            )

    # ------ Fixed/Gripper Camera -> 2D Blind Pick ------
    for observation_mode in ["FO", "PO"]:
        for difficulty in [0.07, 0.15]:
            register(
                id=f"{observation_mode}FixedGripper2DBlind{int(difficulty*100)}cmPick-v0",
                entry_point="gymnasium_robotics.envs.fetch.blind_pick:FetchBlindPickEnv",
                max_episode_steps=100,
                disable_env_checker=True,
                kwargs={
                    "camera_names": ["camera_front", "camera_side", "gripper_camera_rgb"] if "FO" in observation_mode else None,
                    "width": 64,
                    "height": 64,
                    "render_mode": "rgb_array",
                    "obj_range": difficulty,
                },
            )

    # ------ 32x32 Fixed/Gripper Camera -> 2D Blind Pick ------
    for observation_mode in ["FO", "PO"]:
        for difficulty in [0.07, 0.15]:
            register(
                id=f"{observation_mode}32pxFixedGripper2DBlind{int(difficulty*100)}cmPick-v0",
                entry_point="gymnasium_robotics.envs.fetch.blind_pick:FetchBlindPickEnv",
                max_episode_steps=100,
                disable_env_checker=True,
                kwargs={
                    "camera_names": ["camera_front", "camera_side", "gripper_camera_rgb"] if "FO" in observation_mode else None,
                    "width": 32,
                    "height": 32,
                    "render_mode": "rgb_array",
                    "obj_range": difficulty,
                },
            )

    # ------ HORA 32x32 Fixed/Gripper Camera -> 2D Blind Pick ------
    register(
        id=f"HORABlind7cmPick-v0",
        entry_point="gymnasium_robotics.envs.fetch.blind_pick_hora:FetchBlindPickEnv",
        max_episode_steps=100,
        disable_env_checker=True,
        kwargs={
            "render_mode": "rgb_array",
            "obj_range": 0.07,
            "include_obj_state": True,
        },
    )

    # ------ 2D Blind Picking, Object State sanity check ------
    for difficulty in [0.07, 0.15]:
        register(
            id=f"State2DBlind{int(difficulty*100)}cmPick-v0",
            entry_point="gymnasium_robotics.envs.fetch.blind_pick:FetchBlindPickEnv",
            max_episode_steps=100,
            disable_env_checker=True,
            kwargs={
                "camera_names": ["external_camera_0"],
                "width": 64,
                "height": 64,
                "render_mode": "rgb_array",
                "include_obj_state": True,
                "obj_range": difficulty,
            },
        )

    # ------ Pick and Place, State sanity check ------
    register(
        id=f"State7cmPick5cmPlace-v0",
        entry_point="gymnasium_robotics.envs.fetch.blind_pick_place:FetchBlindPickPlaceEnv",
        max_episode_steps=100,
        disable_env_checker=True,
        kwargs={
            "camera_names": ["external_camera_0"],
            "width": 64,
            "height": 64,
            "render_mode": "rgb_array",
            "include_obj_state": True,
            "include_bin_state": True,
        },
    )

    # Pick and Place, Fixed Cam -> Hand Cam Policy
    for difficulty in [0.07]:
        register(
            id=f"FixedToHand{int(difficulty*100)}cmPick5cmPlace-v0",
            entry_point="gymnasium_robotics.envs.fetch.blind_pick_place:FetchBlindPickPlaceEnv",
            max_episode_steps=100,
            disable_env_checker=True,
            kwargs={
                "camera_names": ["camera_behind", "camera_side", "gripper_camera_rgb"],
                "width": 64,
                "height": 64,
                "render_mode": "rgb_array",
                "obj_range": difficulty,
            },
        )
    register(
        id=f"POFixedToHand7cmPick5cmPlace-v0",
        entry_point="gymnasium_robotics.envs.fetch.blind_pick_place:FetchBlindPickPlaceEnv",
        max_episode_steps=100,
        disable_env_checker=True,
        kwargs={
            "camera_names": ["gripper_camera_rgb"],
            "width": 64,
            "height": 64,
            "render_mode": "rgb_array",
            "obj_range": 0.07,
        },
    )
    register(
        id=f"HORAFixedToHand7cmPick5cmPlace-v0",
        entry_point="gymnasium_robotics.envs.fetch.blind_pick_place_hora:FetchBlindPickPlaceEnv",
        max_episode_steps=100,
        disable_env_checker=True,
        kwargs={
            "camera_names": ["gripper_camera_rgb"],
            "width": 32,
            "height": 32,
            "render_mode": "rgb_array",
            "obj_range": 0.07,
        },
    )

    # Pick and Place, Hand Cam -> Fixed Cam Policy
    for difficulty in [0.07]:
        register(
            id=f"HandToFixed{int(difficulty*100)}cmPick5cmPlace-v0",
            entry_point="gymnasium_robotics.envs.fetch.blind_pick_place:FetchBlindPickPlaceEnv",
            max_episode_steps=100,
            disable_env_checker=True,
            kwargs={
                "camera_names": ["camera_front", "gripper_camera_rgb"],
                "width": 64,
                "height": 64,
                "render_mode": "rgb_array",
                "obj_range": difficulty,
            },
        )
    register(
        id=f"POHandToFixed7cmPick5cmPlace-v0",
        entry_point="gymnasium_robotics.envs.fetch.blind_pick_place:FetchBlindPickPlaceEnv",
        max_episode_steps=100,
        disable_env_checker=True,
        kwargs={
            "camera_names": ["camera_front"],
            "width": 64,
            "height": 64,
            "render_mode": "rgb_array",
            "obj_range": 0.07,
        },
    )
    register(
        id=f"HORAHandToFixed7cmPick5cmPlace-v0",
        entry_point="gymnasium_robotics.envs.fetch.blind_pick_place_hora:FetchBlindPickPlaceEnv",
        max_episode_steps=100,
        disable_env_checker=True,
        kwargs={
            "camera_names": ["camera_front"],
            "width": 32,
            "height": 32,
            "render_mode": "rgb_array",
            "obj_range": 0.07,
        },
    )

    for observation_mode in ["FO", "PO", "DepthFO", "DepthPO"]:
        register(
            id=f"{observation_mode}OccludedPickPlace-v0",
            entry_point="gymnasium_robotics.envs.fetch.occluded_pick_place:FetchOccludedPickPlaceEnv",
            max_episode_steps=100,
            disable_env_checker=True,
            kwargs={
                "camera_names": ["external_camera_0", "behind_camera"] if "FO" in observation_mode else ["external_camera_0"],
                "width": 64,
                "height": 64,
                "render_mode": "depth_array" if "depth" in observation_mode.lower() else "rgb_array",
            },
        )

    for observation_mode in ["FO", "PO", "DepthFO", "DepthPO"]:
        register(
            id=f"{observation_mode}OccludedPickPlaceRecessed-v0",
            entry_point="gymnasium_robotics.envs.fetch.occluded_pick_place_recessed:FetchOccludedPickPlaceRecessedEnv",
            max_episode_steps=100,
            disable_env_checker=True,
            kwargs={
                "camera_names": ["behind_camera"] if "FO" in observation_mode else [],
                "width": 64,
                "height": 64,
                "render_mode": "depth_array" if "depth" in observation_mode.lower() else "rgb_array",
                "obj_grip_rew_weight": 0.1,
                "obj_goal_rew_weight": 1
            },
        )

    for observation_mode in ["FO", "PO", "DepthFO", "DepthPO"]:
        register(
            id=f"{observation_mode}OccludedPickPlaceRecessed2Goal-v0",
            entry_point="gymnasium_robotics.envs.fetch.occluded_pick_place_recessed_2goal:FetchOccludedPickPlaceRecessed2GoalEnv",
            max_episode_steps=100,
            disable_env_checker=True,
            kwargs={
                "camera_names": ["external_camera_0", "behind_camera"] if "FO" in observation_mode else ["external_camera_0"],
                "width": 64,
                "height": 64,
                "render_mode": "depth_array" if "depth" in observation_mode.lower() else "rgb_array",
                "obj_grip_rew_weight": 0.1,
                "obj_goal_rew_weight": 1
            },
        )

__version__ = "1.2.1"


try:
    import sys

    from farama_notifications import notifications

    if (
        "gymnasium_robotics" in notifications
        and __version__ in notifications["gymnasium_robotics"]
    ):
        print(notifications["gymnasium_robotics"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass
