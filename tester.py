"""
Tester for MD-CVRP
"""
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import logging
import sys

from torch.utils.data import DataLoader
from tensordict import TensorDict
import torch

from envs import MDCVRPEnv
from models import Actor, Critic


NOW = datetime.strftime(datetime.now(), "%y%m%d-%H%M%S")


class MDCVRPTester:
    def __init__(
        self, env_params: dict[str, Any], model_params: dict[str, Any], tester_params: dict[str, Any]
    ) -> None:
        ### params ###
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        ### update common params ###
        self.env_params.update({"device": self.tester_params["device"]})
        self.device = torch.device(self.tester_params["device"])

        ### Paths & Loggers ###
        prob_setting = f"N{self.env_params['n_custs']}_M{self.env_params['n_agents']}"
        exp_name = f"{self.tester_params['exp_name']}_{NOW}"

        ### Env ###
        self.env, self.test_dataset = self.setup_test()

        ### Load Model ###
        with open(Path(self.model_params["model_ckpt_path"]).parent.parent / "params.json", "r") as f:
            train_params = json.load(f)

        self.actor = Actor(env=self.env, **train_params["model_params"]["actor_params"]).to(self.device)
        self.critic = Critic(env=self.env, **train_params["model_params"]["critic_params"]).to(self.device)

        self.actor.load_state_dict(torch.load(self.model_params["model_ckpt_path"])["actor"])
        self.critic.load_state_dict(torch.load(self.model_params["model_ckpt_path"])["critic"])

        # Paths
        self.result_dir = Path(self.tester_params["result_dir"]) / prob_setting / exp_name
        self.figure_dir = self.result_dir / "figures"

        for _dir in [self.result_dir, self.figure_dir]:
            _dir.mkdir(parents=True, exist_ok=True)

        # Loggers
        self.logger, self.file_logger = logging.getLogger("stdout_logger"), logging.getLogger("file_logger")
        self.logger.setLevel(logging.INFO)
        self.file_logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.file_logger.addHandler(logging.FileHandler(self.result_dir / "stdout.log"))

        # save params
        with open(self.result_dir / "params.json", "w") as f:
            json.dump(
                {
                    "env_params": self.env_params,
                    "model_params": self.model_params,
                    "tester_params": self.tester_params,
                },
                f,
                indent=4,
            )

    def setup_test(self) -> tuple[MDCVRPEnv, TensorDict]:
        """
        Create test environment and make test dataset
        If testset_path is given, env is created based on it
        """
        if self.env_params["testset_path"] is not None:
            env, td = MDCVRPEnv.from_csv(self.env_params["testset_path"], device=self.env_params["device"])
        else:
            env = MDCVRPEnv(**self.env_params)
            td = env.generate_data(self.env_params["test_n_samples"])
        return env, td

    def test(self) -> None:
        self.logger.info("Start Testing...")

        self.actor.phase = "eval"
        self.actor.eval()
        self.critic.eval()

        test_dataloader = DataLoader(self.test_dataset, batch_size=self.tester_params["batch_size"], collate_fn=lambda x: x)  # type: ignore

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                self.env.reset(batch)  # to take steps with mini-batch
                obs_td, actions, _, rewards = self.actor(batch)
                reward = rewards[:, -1]  # (batch_size,), Now, we use only the last reward

                # to cpu for logging
                obs_td, actions = obs_td.cpu(), actions.cpu()
                # unflatten action
                actions = torch.stack(
                    [
                        actions // (self.env.n_agents + self.env.n_custs),
                        actions % (self.env.n_agents + self.env.n_custs),
                    ],
                    dim=-1,
                )

                for _i, (_td, _actions) in enumerate(zip(obs_td, actions)):
                    instance_idx = batch_idx * self.tester_params["batch_size"] + _i
                    log_msg = f"Instance {instance_idx} Result\n"

                    # print agent-wise results
                    agent_route = [[] for _ in range(self.env.n_agents)]

                    for _agent_idx, _node_idx in list(_actions):
                        _agent_idx, _node_idx = int(_agent_idx.item()), int(_node_idx.item())
                        _idx = -1 if _node_idx < self.env.n_agents else _node_idx - self.env.n_agents
                        agent_route[_agent_idx].append(_idx)

                    agent_length = _td["cum_length"].cpu().numpy()  # (n_agents,)

                    for _a in range(self.env.n_agents):
                        log_msg += f"Agent {_a} [{agent_length[_a]:.3f}] ::: {' > '.join(map(str, agent_route[_a]))}\n"

                    self.logger.info(log_msg)
                    self.file_logger.info(log_msg)
                    self.env.render(_td, _actions, save_path=self.figure_dir / f"instance{instance_idx}.png")


if __name__ == "__main__":
    from tester import MDCVRPTester

    file_name = "example"

    env_params = {
        ### When test with pre-generated testset ###
        "testset_path": f"data/test/N20_M2/{file_name}.csv",
        ### When test with randomly generated testset ###
        "n_custs": 20,
        "n_agents": 2,
        "min_loc": 0,
        "max_loc": 1,
        "min_demand": 1,
        "max_demand": 9,
        "vehicle_capacity": None,
        "one_by_one": True,
        "test_n_samples": 0,
    }

    model_params = {
        "model_ckpt_path": "results/N20_M2/debug_230916-135710/checkpoints/epoch40.pth",
    }

    tester_params = {
        ### CPU or GPU ###
        "device": "cpu",
        ### Testing ###
        "batch_size": 256,
        ### Logging and Saving ###
        "result_dir": "test_results",
        "exp_name": file_name,
    }

    tester = MDCVRPTester(env_params, model_params, tester_params)
    tester.test()
