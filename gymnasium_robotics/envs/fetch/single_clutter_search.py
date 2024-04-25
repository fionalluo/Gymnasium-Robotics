from pathlib import Path
from gymnasium_robotics.envs.fetch.blind_pick import FetchBlindPickEnv

MODEL_PATH = Path(__file__).parent.parent.resolve() / 'assets' / 'fetch' / 'single_clutter_search.xml'
assert MODEL_PATH.exists(), f'Model path: {MODEL_PATH} does not exist'

class SingleClutterSearch(FetchBlindPickEnv):
    def __init__(self, model_path=str(MODEL_PATH), *args, **kwargs):
        super().__init__(*args, model_path=model_path, **kwargs)

    def _reset_sim(self):
        super()._reset_sim()
        # move distractor object on top of target
        target_object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
        distractor_qpos = target_object_qpos.copy()
        distractor_qpos[2] += 0.01
        self._utils.set_joint_qpos(self.model, self.data, "object1:joint", distractor_qpos)
        self._mujoco.mj_forward(self.model, self.data)
        return True


if __name__ == '__main__':
    env = SingleClutterSearch(render_mode='human', obj_range=0.05)
    while True:
        env.reset()