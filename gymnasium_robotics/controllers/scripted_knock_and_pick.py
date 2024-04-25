from enum import Enum

import numpy as np
import numpy.typing as npt


class PickPolicyState(Enum):
        KNOCK = 'KNOCK'
        GOTO = 'GOTO'
        GRASP = 'GRASP'

class KnockAndPickPolicy:
    def __init__(self, goal: npt.NDArray, speed=0.5, vector_env=True, verbose=False):
        self.goal = goal
        self.speed = speed
        self.vector_env = vector_env
        self.verbose = verbose

        # State Flags
        self.has_knocked = False
        self.has_grasped = False
        self.reset()


    def reset(self):
        self.state = PickPolicyState.GOTO
        self.has_knocked = False
        self.has_grasped = False
        self.start_obj_state = None


    def __call__(self, obs_dict, state=None):
        action = self.advance_state(obs_dict)
        return {'action': action}, state


    def advance_state(self, obs):
        single_env_obs = self._unpack_obs(obs)
        if self.start_obj_state is None:
            self.start_obj_state = single_env_obs['obj_state'][:3]
        target_pos = self._get_target_pos(single_env_obs)
        if self._met_state_goal(target_pos, single_env_obs):
            self.state = self._get_next_state()  # Advance state if we met prior state's goal
            target_pos = self._get_target_pos(single_env_obs)

        # Get target position and grip
        pos_delta = target_pos - single_env_obs['robot_state'][:3]
        grip_delta = np.array([-1], dtype=np.float32) if self.state == PickPolicyState.GRASP or self.has_grasped else np.array([1], dtype=np.float32)
        action = np.concatenate([self.speed * pos_delta/(np.max(np.abs(pos_delta)+1e-4)), grip_delta], axis=-1)
        return self._pack_action(action)
    
    def _unpack_obs(self, obs):
        return {k : v[0] for k, v in obs.items()} if self.vector_env else obs
    
    def _pack_action(self, action):
        return np.clip(np.array([action], dtype=action.dtype) if self.vector_env else action, -1, 1)
    
    def _get_next_state(self):
        next_state = None
        if self.state == PickPolicyState.GOTO:
            if not self.has_knocked:
                next_state = PickPolicyState.KNOCK
            elif not self.has_grasped:
                next_state = PickPolicyState.GRASP
        elif self.state == PickPolicyState.KNOCK:
            self.has_knocked = True
            next_state = PickPolicyState.GOTO
        elif self.state == PickPolicyState.GRASP:
            self.has_grasped = True
            next_state = PickPolicyState.GOTO
        if self.verbose:
            print(f'{self.state} --> {next_state}')
        
        if next_state is None:
            raise RuntimeError(f'Unknown state execution sequence: {self.state=}, {self.has_knocked=}, {self.has_grasped=}')
        
        return next_state
        

    def _get_target_pos(self, obs):
        if self.state == PickPolicyState.GOTO:
            if not self.has_knocked:
                mult = 1 if self.start_obj_state[1] <= self.goal[1] else -1
                return self.start_obj_state + np.array([0.0, mult*0.13, 0.03], dtype=np.float32)  # some are on other side of object
            elif not self.has_grasped:
                return obs['obj_state'][:3]
            else:
                return self.goal
        elif self.state == PickPolicyState.KNOCK:
            mult = 1 if self.start_obj_state[1] <= self.goal[1] else -1
            return self.start_obj_state + np.array([0.0, -mult*0.08, 0.03], dtype=np.float32)  # some are on other side of object
        elif self.state == PickPolicyState.GRASP:
            return obs['robot_state'][:3]  # same position


    def _met_state_goal(self, goal_pos, obs):
        return {
            PickPolicyState.GOTO: lambda goal_pos, obs : np.linalg.norm(goal_pos - obs['robot_state'][:3]) < 0.02,
            PickPolicyState.KNOCK: lambda goal_pos, obs : np.linalg.norm(goal_pos - obs['robot_state'][:3]) < 0.02,
            PickPolicyState.GRASP: lambda goal_pos, obs : obs['touch'].all()
        }[self.state](goal_pos, obs)

if __name__ == '__main__':
    MULTI_OBJ_ENV = True
    TEST_CONTROLLER = False
    COLLECT_DATA = True

    if TEST_CONTROLLER:
        from gymnasium_robotics.envs.fetch.single_clutter_search import SingleClutterSearch
        from gymnasium_robotics.envs.fetch.clutter_search import FetchClutterSearchEnv
        

        eval_episodes = 3
        print(f'Testing Controller for {eval_episodes} episodes...')
        kwargs= {
            "render_mode": "human",
            "camera_names": ["camera_front"],
            "width": 64,
            "height": 64,
            "include_obj_state": True,
            "obj_range": 0.05,
        }

        controller = KnockAndPickPolicy(np.array([1.33, 0.75, 0.60]), vector_env=False, verbose=False)
        env = FetchClutterSearchEnv(**kwargs) if MULTI_OBJ_ENV else SingleClutterSearch(**kwargs)
        for _ in range(eval_episodes):
            obs, _ = env.reset()
            controller.reset()
            done = False
            while not done:
                action = controller(obs)[0]['action']
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated


    if COLLECT_DATA:
        from pathlib import Path
        import dreamerv3.embodied as embodied
        from dreamerv3.embodied.replay.log_replay_wrapper import FromGymnasiumLogReplayDriver

        collection_episodes = 500
        logdir = Path('./logdir/multi_clutter_search_demos')
        env_str = 'ClutterSearch2x2-v0' if MULTI_OBJ_ENV else 'SingleClutterSearch0.1cm-v0'

        print(f'Collecting {collection_episodes} episodes...')
        controller = KnockAndPickPolicy(np.array([1.33, 0.75, 0.60]), vector_env=True)
        logger = embodied.Logger(embodied.Counter(), [embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score')])
        on_eps = [lambda *args, **kwargs: controller.reset()]
        driver = FromGymnasiumLogReplayDriver(env_str, dir=logdir, logger=logger, chunks=64, on_episode_fns=on_eps)

        driver.run(controller, steps=0, episodes=collection_episodes)
