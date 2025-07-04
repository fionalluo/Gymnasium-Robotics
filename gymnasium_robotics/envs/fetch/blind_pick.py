"""
This script defines the FetchBlindPick environment.

The task is for the Fetch robot to pick up a cube from a table and lift it to a target position in the air.
The robot may not have a camera providing a direct view of the object,
and instead must rely on proprioception and touch sensors. The starting position of the cube is randomized
in a rectangular region on the table. 
`obj_range` is the maximum distance from the center of the region that the cube can be placed. Higher values of `obj_range`
mean a more difficult task.

Observation Space:
The observation is a dictionary with the following keys:
- `robot_state`: (10,) numpy array containing gripper position, velocity, and joint states.
- `touch`: (2,) numpy array representing boolean touch sensor data for the left and right fingers.
- `<camera_name>` (optional): (H, W, 3) numpy array for RGB image from a camera, if specified.
    Can be: gripper_camera_rgb, camera_front, camera_side
- `obj_state` (optional): (3,) numpy array for the object's position.

Action Space:
The action space is a (4,) numpy array:
- action[0]: Change in gripper's x position.
- action[1]: Change in gripper's y position.
- action[2]: Change in gripper's z position.
- action[3]: Gripper open/close state.

Rewards:
The environment provides a dense reward structure:
- Reaching reward: Based on the distance between the gripper and the cube.
- Grasping reward: A bonus for making contact with the cube with both fingers.
- Picking reward: Based on the distance between the grasped cube and the goal position.
A large success bonus is given when the cube reaches the goal.
"""
import os

import numpy as np
import mujoco

from gymnasium import spaces 
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, goal_distance

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "blind_pick.xml")


class FetchBlindPickEnv(MujocoFetchEnv, EzPickle):
    metadata = {"render_modes": ["rgb_array", "depth_array"], 'render_fps': 25}
    render_mode = "rgb_array"
    def __init__(self, camera_names=None, reward_type="dense", obj_range=0.07, 
                #  include_obj_state=False, 
                 include_obj_state=True, 
                 model_path=MODEL_XML_PATH, 
                 max_episode_limit=None,
                 **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            'object0:joint': [1.33, 0.75, 0.42, 1., 0., 0., 0.],
        }
        self.camera_names = camera_names if camera_names is not None else []
        workspace_min=np.array([1.1, 0.44, 0.42])
        workspace_max=np.array([1.5, 1.05, 0.7])
        
        self.episodes_so_far = 0
        self.max_episode_limit = max_episode_limit
        self.total_reward = 0
        self.last_10_successes = []
        self.curriculum = 0

        self.workspace_min = workspace_min
        self.workspace_max = workspace_max
        self.initial_qpos = initial_qpos
        MujocoFetchEnv.__init__(
            self,
            model_path=model_path,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=obj_range,
            target_range=0.0,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        print("reward_type:", reward_type)
        self.cube_body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "object0"
        )
        # consists of images and proprioception.
        _obs_space = {}
        if isinstance(camera_names, list) and len(camera_names) > 0:
            for c in camera_names:
                _obs_space[c] = spaces.Box(
                        0, 255, shape=(self.height, self.width, 3), dtype="uint8"
                    ) if self.render_mode == "rgb_array" else \
                    spaces.Box(
                        0, np.inf, shape=(self.height, self.width, 1), dtype="float32"
                    )
        _obs_space["robot_state"] = spaces.Box(-np.inf, np.inf, shape=(10,), dtype="float32")
        _obs_space["touch"] = spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float32")
        self.include_obj_state = include_obj_state
        if include_obj_state:
            _obs_space["obj_state"] = spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32")

        self.observation_space = spaces.Dict(_obs_space)
        EzPickle.__init__(self, camera_names=camera_names, image_size=32, reward_type=reward_type, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.33, 0.75, 0.60])
        return goal.copy()

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = [1.3, 0.75]

            # Add curriculum for object (linearly increase obj_range)
            if self.max_episode_limit:
                t = self.episodes_so_far / self.max_episode_limit
                obj_range_curriculum = max(min(self.obj_range * t, self.obj_range), 0.025)
            else:
                obj_range_curriculum = self.obj_range

            # sample in a rectangular region and offset by a random amount
            object_xpos[0] += self.np_random.uniform(-obj_range_curriculum, obj_range_curriculum)
            y_offset = self.np_random.uniform(-obj_range_curriculum, obj_range_curriculum)
            # object_xpos[0] += self.np_random.uniform(-self.obj_range, self.obj_range)
            # y_offset = self.np_random.uniform(-self.obj_range, self.obj_range)
            object_xpos[1] += y_offset
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True
    
    def _get_obs(self):
        obs = {}
        if hasattr(self, "mujoco_renderer"):
            self._render_callback()
            for c in self.camera_names:
                # New Gymnasium MujocoRenderer.render() no longer accepts a
                # `camera_name` argument; instead the camera is selected via
                # the `camera_id` attribute set on the renderer prior to
                # calling `render`. Convert our desired camera *name* to an
                # integer ID and assign it before rendering.

                try:
                    cam_id = self._mujoco.mj_name2id(
                        self.model, self._mujoco.mjtObj.mjOBJ_CAMERA, c
                    )
                except Exception:
                    cam_id = -1  # fallback to default track camera

                # Update renderer to use this camera ID, then render.
                self.mujoco_renderer.camera_id = cam_id
                img = self.mujoco_renderer.render(self.render_mode)
                obs[c] = img[:,:,None] if self.render_mode == 'depth_array' else img

            touch_left_finger = False
            touch_right_finger = False
            obj = "object0"
            l_finger_geom_id = self.model.geom("robot0:l_gripper_finger_link").id
            r_finger_geom_id = self.model.geom("robot0:r_gripper_finger_link").id
            for j in range(self.data.ncon):
                c = self.data.contact[j]
                body1 = self.model.geom_bodyid[c.geom1]
                body2 = self.model.geom_bodyid[c.geom2]
                body1_name = self.model.body(body1).name
                body2_name = self.model.body(body2).name

                if c.geom1 == l_finger_geom_id and body2_name == obj:
                    touch_left_finger = True
                if c.geom2 == l_finger_geom_id and body1_name == obj:
                    touch_left_finger = True

                if c.geom1 == r_finger_geom_id and body2_name == obj:
                    touch_right_finger = True
                if c.geom2 == r_finger_geom_id and body1_name == obj:
                    touch_right_finger = True

            obs["touch"] = np.array([int(touch_left_finger), int(touch_right_finger)]).astype(np.float32)

            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

            dt = self.n_substeps * self.model.opt.timestep
            grip_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
            )

            robot_qpos, robot_qvel = self._utils.robot_get_obs(
                self.model, self.data, self._model_names.joint_names
            )
            gripper_state = robot_qpos[-2:]
            gripper_vel = robot_qvel[-2:] * dt # change to a scalar if the gripper is made symmetric
            
            obs["robot_state"] = np.concatenate([grip_pos, grip_velp, gripper_state, gripper_vel]).astype(np.float32)
            if self.include_obj_state:
                obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0").copy()
                obs["obj_state"] = obj0_pos.astype(np.float32)

        else:
            # BaseRobotEnv has called _get_obs to determine observation space dims but mujoco renderer has not been initialized yet.
            # in this case, return an obs dict with arbitrary values for each ey
            # since observation space will be overwritten later.
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8) if self.render_mode == "rgb_array" \
                else np.zeros((self.height, self.width, 1), dtype=np.float32)
            obs["achieved_goal"] = obs["observation"] = img
        return obs

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # check if action is out of bounds
        curr_eef_state = self._utils.get_site_xpos(self.model, self.data, 'robot0:grip')
        next_eef_state = curr_eef_state + (action[:3] * 0.05)

        next_eef_state = np.clip(next_eef_state, self.workspace_min, self.workspace_max)
        clipped_ac = (next_eef_state - curr_eef_state) / 0.05
        action[:3] = clipped_ac

        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        info = {
            "is_success": self._is_success(obj0_pos, self.goal),
        }

        terminated = goal_distance(obj0_pos, self.goal) < 0.05
        # handled by time limit wrapper.
        truncated = self.compute_truncated(obj0_pos, self.goal, info)

        # reward = self.compute_reward(obj0_pos, self.goal, info)
        # success bonus
        reward = 0
        if terminated:
            # print("success phase")
            reward = 300
        else:
            if self.reward_type == "dense":
                dist = np.linalg.norm(curr_eef_state - obj0_pos)
                reaching_reward = 1 - np.tanh(10.0 * dist)
                reward += reaching_reward
                # msg = "reaching phase"

                # grasping reward
                if obs["touch"].all():
                    reward += 0.25
                    dist = np.linalg.norm(self.goal - obj0_pos)
                    picking_reward = 1 - np.tanh(10.0 * dist)
                    reward += picking_reward
                #     msg = "picking phase"
                # print(msg)

        self.total_reward += reward
        return obs, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):  
        # def print_to_file(file_path, message):
        #     with open(file_path, 'a') as file:
        #         file.write(message + '\n')
        
        # message = f"In reset: max episode limit {self.episodes_so_far}/{self.max_episode_limit}"
        # print_to_file('/home/harsh/fiona/dreamerv3/test_log.txt', message)

        # removed super.reset call
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()

        # Check if we need to increase the curricula
        self.last_10_successes.append(self.total_reward >= 300)
        self.total_reward = 0
        if len(self.last_10_successes) > 10:
            self.last_10_successes = self.last_10_successes[1:]
        if sum(self.last_10_successes) >= 9:
            self.curriculum += 1

        noise = 0.04

        def open_gripper():
            action = np.array([0.0, 0.0, 0.0, 1.0])
            action = np.clip(action, self.action_space.low, self.action_space.high)
            curr_eef_state = self._utils.get_site_xpos(self.model, self.data, 'robot0:grip')
            next_eef_state = curr_eef_state + (action[:3] * 0.05)
            next_eef_state = np.clip(next_eef_state, self.workspace_min, self.workspace_max)
            clipped_ac = (next_eef_state - curr_eef_state) / 0.05
            action[:3] = clipped_ac
            self._set_action(action)
            for _ in range(10):
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        def close_gripper():
            action = np.array([0.0, 0.0, 0.0, -1.0])
            action = np.clip(action, self.action_space.low, self.action_space.high)
            curr_eef_state = self._utils.get_site_xpos(self.model, self.data, 'robot0:grip')
            next_eef_state = curr_eef_state + (action[:3] * 0.05)
            next_eef_state = np.clip(next_eef_state, self.workspace_min, self.workspace_max)
            clipped_ac = (next_eef_state - curr_eef_state) / 0.05
            action[:3] = clipped_ac
            self._set_action(action)
            for _ in range(10):
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        def move_above_cube():
            gripper_target = self._utils.get_site_xpos(self.model, self.data, "object0") + np.array([0, 0, 0.1])
            gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
            self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
            self._utils.set_mocap_quat(
                self.model, self.data, "robot0:mocap", gripper_rotation
            )
            for _ in range(10):
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        def move_down():
            lower_gripper_target = self._utils.get_site_xpos(self.model, self.data, 'robot0:grip') - np.array([0, 0, 0.1])
            self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", lower_gripper_target)
            for _ in range(10):
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        def move_to_goal(t: float = 1):
            # Move the gripper a scaled distance toward the goal. Ratio = amount to move to goal
            start_gripper_target = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            end_gripper_target = self.goal
            gripper_target = start_gripper_target + t * (end_gripper_target - start_gripper_target)
            self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
            for _ in range(10):
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        def move_to_cube(t: float = 1):
            # Move the gripper a scaled distance toward the goal. Ratio = amount to move to goal
            start_gripper_target = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            end_gripper_target = self._utils.get_site_xpos(self.model, self.data, "object0") + np.array([0, 0, 0.1])
            gripper_target = start_gripper_target + t * (end_gripper_target - start_gripper_target)
            self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
            for _ in range(10):
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        def add_noise(noise: float = 0.02):
            gripper_target = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            gripper_target[0] += self.np_random.uniform(-noise, noise)
            gripper_target[1] += self.np_random.uniform(-noise, noise)
            gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
            self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
            self._utils.set_mocap_quat(
                self.model, self.data, "robot0:mocap", gripper_rotation
            )
            for _ in range(10):
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        
        if self.max_episode_limit:
            
            self.episodes_so_far += 1

            # t = self.episodes_so_far / self.max_episode_limit

            # move_to_cube(max(0, 1 - t))
            add_noise(0.02)
        
        elif self.max_episode_limit == 0:
            # Handle 0 case. None is NOT covered by case, so if None, fixed position WITHOUT noise
            add_noise(0.02)
        
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, {}

    def close(self):
        pass

if __name__ == "__main__":
    
    kwargs= {
        "camera_names": ["external_camera_0"],
        "width": 64,
        "height": 64,
        "include_obj_state": True,
        "obj_range": 0.07,
        "max_episode_limit": 500,
    }
    env = FetchBlindPickEnv(render_mode="human", **kwargs)
    while True:
        env.reset()
    
    # import omegaconf
    # import hydra
    # import torch
    # import torchvision.transforms as T
    # import numpy as np
    # from PIL import Image

    # from vip import load_vip

    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"

    # vip = load_vip()
    # vip.eval()
    # vip.to(device)

    # ## DEFINE PREPROCESSING
    # transforms = T.Compose([T.Resize(256),
    #     T.CenterCrop(224),
    #     T.ToTensor()]) # ToTensor() divides by 255

    # ## ENCODE IMAGE
    # image = np.random.randint(0, 255, (500, 500, 3))
    # preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
    # preprocessed_image.to(device) 
    # with torch.no_grad():
    #     embedding = vip(preprocessed_image * 255.0) ## vip expects image input to be [0-255]
    # print(embedding.shape) # [1, 1024]

    import imageio
    cam_keys = ["camera_front", "camera_front_2"]
    # env = FetchBlindPickEnv(cam_keys, "dense", render_mode="depth_array", width=32, height=32, obj_range=0.001)

    env = FetchBlindPickEnv(render_mode="human", **kwargs)

    # imgs = []
    # obs, _ = env.reset()



    # def process_depth(depth):
    #     # depth -= depth.min()
    #     # depth /= 2*depth[depth <= 1].mean()
    #     # pixels = 255*np.clip(depth, 0, 1)
    #     # pixels = pixels.astype(np.uint8)
    #     # return pixels
    #     return depth
    # for _ in range(100):
    #     obs,_ = env.reset()
    #     imgs.append(np.concatenate([obs['camera_side'], obs['camera_front'], obs['gripper_camera_rgb']], axis=1))
    #     # for i in range(10):
    #     #     obs, *_ = env.step(env.action_space.sample())
    #     #     imgs.append(np.concatenate([obs['camera_side'], obs['camera_front'], obs['gripper_camera_rgb']], axis=1))
    # imageio.mimwrite("test.gif", imgs)
    from collections import defaultdict
    demo = defaultdict(list)
    while True:
        obs, _ = env.reset()
        for k in obs.keys():
            if k in cam_keys:
                demo[k].append(obs[k])
        # open the gripper and descend
        for i in range(10):
            obs, rew, term, trunc, info = env.step(np.array([-0.1, 0.0, -1, 1.0]))
            for k in obs.keys():
                if k in cam_keys:
                    demo[k].append(obs[k])
            # print(rew)
        # close gripper
        for i in range(10):
            obs, rew, term, trunc, info= env.step(np.array([0,0,0.0,-1.0]))
            for k in obs.keys():
                if k in cam_keys:
                    demo[k].append(obs[k])
            print(rew)
        # lift up cube
        for i in range(10):
            obs, rew, term, trunc, info = env.step(np.array([0,0,1.0,-1.0]))
            for k in obs.keys():
                if k in cam_keys:
                    demo[k].append(obs[k])
            print(rew)
            if term:
                # for k in cam_keys:
                #     imageio.imwrite(f'blindpick_final_{k}.png', obs[k])
                break

    # save each key as a mp4 with imageio                
    for k, v in demo.items():
        imageio.mimwrite(f"{k}.mp4", v)

    # import ipdb; ipdb.set_trace()