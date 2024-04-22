from enum import Enum

import numpy as np
import numpy.typing as npt


class PickPolicyState(Enum):
        APPROACH = 'APPROACH'
        GRASP = 'GRASP'
        LIFT = 'LIFT'

class RandomPickPolicy:
    def __init__(self, goal: npt.NDArray, vector_env=True, verbose=False):
        self.goal = goal
        self.vector_env = vector_env
        self.verbose = verbose
        self.reset()


    def reset(self):
        self.state = PickPolicyState.APPROACH


    def __call__(self, obs_dict, state=None):
        action = self.advance_state(obs_dict)
        return {'action': action}, state


    def advance_state(self, obs):
        next_state = self.state  # default to current state
        pos_delta = np.zeros((1,3), dtype=np.float32) if self.vector_env else np.zeros((3,), dtype=np.float32)
        grip_delta = np.zeros((1,1), dtype=np.float32) if self.vector_env else np.zeros((1,), dtype=np.float32)

        # Approach Target Pick Location Until W/In Threshold
        if self.state == PickPolicyState.APPROACH:
            curr_pos = obs['robot_state'][:,:3] if self.vector_env else obs['robot_state'][:3]
            obj_state = obs['obj_state']
            pos_delta = obj_state - curr_pos
            grip_delta = np.array([[1]], dtype=np.float32) if self.vector_env else np.array([1], dtype=np.float32)
            if np.linalg.norm(pos_delta) < 0.02:
                next_state = PickPolicyState.GRASP

        elif self.state == PickPolicyState.GRASP:
            grip_delta = np.array([[-1]], dtype=np.float32) if self.vector_env else np.array([-1], dtype=np.float32)
            if obs['touch'].all():
                next_state = PickPolicyState.LIFT

        elif self.state == PickPolicyState.LIFT:
            grip_delta = np.array([[-1]], dtype=np.float32) if self.vector_env else np.array([-1], dtype=np.float32)
            curr_pos = obs['robot_state'][:,:3] if self.vector_env else obs['robot_state'][:3]
            pos_delta = self.goal - curr_pos
            if np.linalg.norm(pos_delta) < 0.02:  # w/in 2cm
                next_state = PickPolicyState.APPROACH

        if self.verbose and self.state != next_state:
            print(f'{self.state.value} --> {next_state.value}')

        self.state = next_state
        action = np.concatenate([pos_delta/(np.max(np.abs(pos_delta)+1e-4)), grip_delta], axis=-1)
        action = np.clip(action, -1, 1)
        return action

if __name__ == '__main__':
    TEST_CONTROLLER = False
    COLLECT_DATA = True

    if TEST_CONTROLLER:
        from gymnasium_robotics.envs.fetch.blind_pick import FetchBlindPickEnv

        eval_episodes = 3
        print(f'Testing Controller for {eval_episodes} episodes...')
        kwargs= {
            "camera_names": ["external_camera_0"],
            "width": 64,
            "height": 64,
            "include_obj_state": True,
            "obj_range": 0.001,
        }

        controller = RandomPickPolicy(np.array([1.33, 0.75, 0.60]), vector_env=False)
        env = FetchBlindPickEnv(render_mode="human", **kwargs)
        for _ in range(eval_episodes):
            obs, _ = env.reset()
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
        logdir = Path('./logdir/blind_pick_demos')
        env_str = 'State2DBlind0.1cmPick-v0'

        print(f'Collecting {collection_episodes} episodes...')
        controller = RandomPickPolicy(np.array([[1.33, 0.75, 0.60]]), vector_env=True)
        logger = embodied.Logger(embodied.Counter(), [embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score')])
        driver = FromGymnasiumLogReplayDriver(env_str, dir=logdir, logger=logger, chunks=64)
        driver.run(controller, steps=0, episodes=collection_episodes)

