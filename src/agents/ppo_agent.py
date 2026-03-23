from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
import gymnasium as gym
import os
from typing import Optional


class PPOAgent:
    """
    Thin wrapper around stable-baselines3 PPO.
    Exposes: train(total_timesteps), save(path), load(path), predict(obs).
    Hyperparameters are read from YAML config dict.
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Optional[dict] = None,
        model_path: Optional[str] = None,
        verbose: int = 1,
    ):
        self.cfg = config or {}
        self.model: Optional[PPO] = None
        self._env = env
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            lr = self.cfg.get("learning_rate", 3.0e-4)
            gamma = self.cfg.get("gamma", 0.99)
            clip_range = self.cfg.get("clip_range", 0.2)
            n_steps = self.cfg.get("n_steps", 2048)
            
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=lr,
                gamma=gamma,
                clip_range=clip_range,
                n_steps=n_steps,
                verbose=verbose,
                tensorboard_log=None,
            )
    
    def train(self, total_timesteps: int):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    
    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.model.save(path)
    
    def load(self, path: str):
        self.model = PPO.load(path)
    
    def predict(self, obs):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        action, _ = self.model.predict(obs, deterministic=False)
        return action
    
    def set_env(self, env: gym.Env):
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.set_env(env)
        self._env = env


if __name__ == "__main__":
    print("Testing PPOAgent with mock env...")
    import numpy as np
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environments.ns3_env import Ns3Env
    
    env = Ns3Env(mock=True, config={"delta": 0.0})
    agent = PPOAgent(env, config={
        "learning_rate": 3.0e-4,
        "gamma": 0.99,
        "clip_range": 0.2,
        "n_steps": 128,
    }, verbose=0)
    
    print("Training for 512 steps...")
    agent.train(512)
    
    test_model_path = "/tmp/test_ppo_agent.zip"
    agent.save(test_model_path)
    print(f"Saved to {test_model_path}")
    
    obs, _ = env.reset()
    for _ in range(10):
        action = agent.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    print("PPOAgent tests passed!")
