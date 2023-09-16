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

        ### Env ###
        self.env, self.test_dataset, self.env_params = self.setup_test()

        ### Load Model ###
        with open(Path(self.model_params["model_ckpt_path"]).parent.parent / "params.json", "r") as f:
            train_params = json.load(f)

        self.actor = Actor(env=self.env, **train_params["model_params"]["actor_params"]).to(self.device)
        self.critic = Critic(env=self.env, **train_params["model_params"]["critic_params"]).to(self.device)

        self.actor.load_state_dict(torch.load(self.model_params["model_ckpt_path"])["actor"])
        self.critic.load_state_dict(torch.load(self.model_params["model_ckpt_path"])["critic"])

        ### Paths & Loggers ###
        prob_setting = f"N{self.env_params['n_custs']}_M{self.env_params['n_agents']}_D{self.env_params['dimension']}"
        exp_name = f"{NOW}_{self.tester_params['exp_name']}"

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

    def setup_test(self) -> tuple[MDCVRPEnv, TensorDict, dict[str, Any]]:
        """
        Create test environment and make test dataset
        If testset_path is given, env is created based on it
        """
        if self.env_params["testset_path"] is not None:
            env, td, env_params = MDCVRPEnv.from_csv(**self.env_params)
        else:
            env = MDCVRPEnv(**self.env_params)
            td = env.generate_data(self.env_params["test_n_samples"], seed=0)
            env_params = self.env_params
        return env, td, env_params

    def test(self) -> None:
        self.logger.info("Start Testing...")

        self.actor.phase = "eval"
        self.actor.eval()
        self.critic.eval()

        test_dataloader = DataLoader(self.test_dataset, batch_size=self.tester_params["batch_size"], collate_fn=lambda x: x)  # type: ignore

        with torch.no_grad():
            n_success = 0
            for batch_idx, batch in enumerate(test_dataloader):
                self.env.reset(batch)  # to take steps with mini-batch
                obs_td, actions, _, _ = self.actor(batch)

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

                    success = _td["demand"].sum(dim=-1).eq(0).numpy().item()
                    n_success += success
                    log_msg = f"Instance {instance_idx} Result [{'SUCCESS' if success else 'FAILED'}]\n"

                    # print agent-wise results
                    agent_route = [["D"] for _ in range(self.env.n_agents)]

                    for _agent_idx, _node_idx in list(_actions):
                        _agent_idx, _node_idx = int(_agent_idx.item()), int(_node_idx.item())
                        _idx = "D" if _node_idx < self.env.n_agents else str(_node_idx - self.env.n_agents)
                        agent_route[_agent_idx].append(_idx)

                    agent_length = _td["cum_length"].numpy()  # (n_agents,)

                    for _a in range(self.env.n_agents):
                        a_route = agent_route[_a]
                        while a_route[-1] == "D":
                            a_route.pop()
                        a_route.append("D")
                        log_msg += f"Agent {_a} [{agent_length[_a]:.3f}] ::: {' > '.join(agent_route[_a])}\n"

                    self.log(log_msg)
                    self.env.render(_td, _actions, save_path=self.figure_dir / f"instance{instance_idx}.png")

        n_samples = len(self.test_dataset)
        self.log(f"Success Rate: {n_success}/{n_samples} ({100 * n_success / n_samples:.2f}%)")
        self.log(f"Testing Finished!")

    def log(self, msg: str) -> None:
        self.logger.info(msg)
        self.file_logger.info(msg)


if __name__ == "__main__":
    testset_path = "data/test/N20_M2_D3/example.csv"  # None for randomly generated testset
    exp_name = "random" if testset_path is None else Path(testset_path).stem

    env_params = {
        ### When test with pre-generated testset ###
        "testset_path": testset_path,  # None for randomly generated testset
        ### params for randomly generated testset, if testset_path is given, they are ignored ###
        "n_custs": 20,
        "n_agents": 2,
        "dimension": 3,
        "min_loc": 0,
        "max_loc": 1,
        "min_demand": 1,
        "max_demand": 1,
        "vehicle_capacity": 1000,
        "test_n_samples": 100,
        ### Env logic params ###
        "one_by_one": False,
        "no_restart": True,
    }

    model_params = {
        "model_ckpt_path": "results/N20_M2_D3/train_230917-010958/checkpoints/epoch400.pth",
    }

    tester_params = {
        ### CPU or GPU ###
        "device": "cuda",  # "cpu" or "cuda"
        ### Testing ###
        "batch_size": 256,
        ### Logging and Saving ###
        "result_dir": "test_results",
        "exp_name": exp_name,
    }

    tester = MDCVRPTester(env_params, model_params, tester_params)
    tester.test()
