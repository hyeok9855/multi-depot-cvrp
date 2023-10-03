"""
Trainer for MD-CVRP
"""
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import logging
import sys

from tensorboard_logger import Logger as TbLogger
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from envs import MDCVRPEnv
from models import Actor, Critic


NOW = datetime.strftime(datetime.now(), "%y%m%d-%H%M%S")


class MDCVRPTrainer:
    def __init__(
        self, env_params: dict[str, Any], model_params: dict[str, Any], trainer_params: dict[str, Any]
    ) -> None:
        ### params ###
        self.env_params = env_params
        self.model_params = model_params
        self.trainer_params = trainer_params

        ### update common params ###
        self.env_params.update({"device": self.trainer_params["device"]})
        d: int = self.env_params["dimension"]
        self.device = torch.device(self.trainer_params["device"])
        # update input_size params of models
        self.model_params["actor_params"]["loc_encoder_params"].update({"input_size": d + 1})
        self.model_params["actor_params"]["rnn_input_encoder_params"].update({"input_size": 2 * d + 1})
        self.model_params["critic_params"]["loc_encoder_params"].update({"input_size": d + 1})

        ### Env ###
        self.env = MDCVRPEnv(**self.env_params)

        ### Model ###
        self.actor = Actor(env=self.env, **model_params["actor_params"]).to(self.device)
        self.critic = Critic(env=self.env, **model_params["critic_params"]).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), **model_params["actor_optimizer"])
        self.critic_optimizer = Adam(self.critic.parameters(), **model_params["critic_optimizer"])

        ### Paths & Loggers ###
        prob_setting = f"N{self.env_params['n_custs']}_M{self.env_params['n_agents']}_D{self.env_params['dimension']}"
        exp_name = f"{NOW}_{self.trainer_params['exp_name']}"

        # Paths
        self.result_dir = Path(self.trainer_params["result_dir"]) / prob_setting / exp_name
        self.checkpoint_dir = self.result_dir / "checkpoints"
        self.figure_dir = self.result_dir / "figures"

        for _dir in [self.result_dir, self.checkpoint_dir, self.figure_dir]:
            _dir.mkdir(parents=True, exist_ok=True)

        # Loggers
        self.logger, self.file_logger = logging.getLogger("stdout_logger"), logging.getLogger("file_logger")
        self.logger.setLevel(logging.INFO)
        self.file_logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.file_logger.addHandler(logging.FileHandler(self.result_dir / "stdout.log"))

        # tensorboard
        self.tb_logger = None
        self.tb_log_dir = None
        self.use_tensorboard = self.trainer_params["use_tensorboard"]
        if self.use_tensorboard:
            self.tb_log_dir = Path(self.trainer_params["tb_log_dir"]) / prob_setting / exp_name
            self.tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_logger = TbLogger(self.tb_log_dir)
        self.step_per_epoch = (
            self.trainer_params["train_n_samples"] // self.trainer_params["batch_size"]
            if self.trainer_params["train_n_samples"] % self.trainer_params["batch_size"] == 0
            else self.trainer_params["train_n_samples"] // self.trainer_params["batch_size"] + 1
        )

        # save params
        with open(self.result_dir / "params.json", "w") as f:
            json.dump(
                {
                    "env_params": self.env_params,
                    "model_params": self.model_params,
                    "trainer_params": self.trainer_params,
                },
                f,
                indent=4,
            )

    def train(self) -> None:
        self.log("Start training...")

        n_epochs = self.trainer_params["n_epochs"]
        early_stop_metric = -float("inf")
        early_stop_count = 0
        for epoch in range(1, n_epochs + 1):
            tr_reward, tr_length, tr_actor_loss, tr_critic_loss = self.train_epoch(epoch)
            val_reward, val_length, val_actor_loss, val_critic_loss = self.validate_epoch(epoch)

            log_msg = (
                f"Epoch {epoch} TRAIN | "
                f"Reward: {tr_reward:.3f} | "
                f"Length: {tr_length:.3f} | "
                f"Actor Loss: {tr_actor_loss:.3f} | "
                f"Critic Loss: {tr_critic_loss:.3f}\n"
                f"Epoch {epoch} VALID | "
                f"Reward: {val_reward:.3f} | "
                f"Length: {val_length:.3f} | "
                f"Actor Loss: {val_actor_loss:.3f} | "
                f"Critic Loss: {val_critic_loss:.3f}"
            )
            self.log(log_msg)

            if self.use_tensorboard and self.tb_logger is not None:
                self.tb_logger.log_value("TRAIN reward/epoch", tr_reward, step=epoch)
                self.tb_logger.log_value("TRAIN length/epoch", tr_length, step=epoch)
                self.tb_logger.log_value("TRAIN loss/epoch", tr_actor_loss, step=epoch)
                self.tb_logger.log_value("TRAIN critic_loss/epoch", tr_critic_loss, step=epoch)

                self.tb_logger.log_value("VALID reward/epoch", val_reward, step=epoch)
                self.tb_logger.log_value("VALID length/epoch", val_length, step=epoch)
                self.tb_logger.log_value("VALID loss/epoch", val_actor_loss, step=epoch)
                self.tb_logger.log_value("VALID critic_loss/epoch", val_critic_loss, step=epoch)

            if epoch == 1 or epoch % self.trainer_params["save_model_interval"] == 0:
                self.save_model(f"epoch{epoch}")

            new_metric = tr_reward if self.trainer_params["early_stop_metric"] == "reward" else -tr_length
            new_best = new_metric > early_stop_metric

            if new_best:
                self.save_model("best")
                early_stop_metric = new_metric
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count == self.trainer_params["ealry_stop_patience"]:
                self.log(f"Early stopping because the reward has not improved for {early_stop_count} epochs.")
                break

        self.log("Training finished!")

    def train_epoch(self, epoch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.actor.phase = "train"
        self.actor.train()
        self.critic.train()

        train_dataset = self.env.generate_data(self.trainer_params["train_n_samples"], seed=None)
        train_dataloader = DataLoader(train_dataset, batch_size=self.trainer_params["batch_size"], collate_fn=lambda x: x)  # type: ignore

        e_rewards = e_lengths = e_actor_losses = e_critic_losses = torch.empty((0,), device=train_dataset.device)
        for batch_idx, batch in enumerate(tqdm(train_dataloader), 1):
            self.env.reset(batch)  # to take steps with mini-batch

            obs_td, _, log_probs, rewards = self.actor(batch)
            rewards = rewards[:, -1]  # (batch_size,), we use only the last reward. TODO: support intermediate rewards
            lengths = obs_td["cum_length"].sum(dim=1)  # (batch_size,)

            critic_bl = self.critic(batch)  # (batch_size,)
            actor_loss, critic_loss = self.calculate_loss(log_probs, rewards, critic_bl)
            self.optimize_models(actor_loss, critic_loss)

            if self.use_tensorboard and self.tb_logger is not None:
                step = (epoch - 1) * self.step_per_epoch + batch_idx
                self.tb_logger.log_value("TRAIN reward/step", rewards.mean(), step=step)
                self.tb_logger.log_value("TRAIN length/step", lengths.mean(), step=step)
                self.tb_logger.log_value("TRAIN actor_loss/step", actor_loss, step=step)
                self.tb_logger.log_value("TRAIN critic_loss/step", critic_loss, step=step)

            e_rewards = torch.cat([e_rewards, rewards.mean().unsqueeze(0)])
            e_lengths = torch.cat([e_lengths, lengths.mean().unsqueeze(0)])
            e_actor_losses = torch.cat([e_actor_losses, actor_loss.unsqueeze(0)])
            e_critic_losses = torch.cat([e_critic_losses, critic_loss.unsqueeze(0)])

        return e_rewards.mean(), e_lengths.mean(), e_actor_losses.mean(), e_critic_losses.mean()

    def validate_epoch(self, epoch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.actor.phase = "eval"
        self.actor.eval()
        self.critic.eval()

        if self.trainer_params["fix_valid_dataset"]:
            if epoch == 1:
                self.valid_dataset = self.env.generate_data(self.trainer_params["valid_n_samples"], seed=0)
            valid_dataset = self.valid_dataset
        else:
            valid_dataset = self.env.generate_data(self.trainer_params["valid_n_samples"])

        valid_dataloader = DataLoader(valid_dataset, batch_size=self.trainer_params["batch_size"], collate_fn=lambda x: x)  # type: ignore

        e_rewards = e_lengths = e_actor_losses = e_critic_losses = torch.empty((0,), device=valid_dataset.device)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valid_dataloader), 1):
                self.env.reset(batch)  # to take steps with mini-batch
                obs_td, actions, log_probs, rewards = self.actor(batch)
                reward = rewards[:, -1]  # (batch_size,), Now, we use only the last reward
                lengths = obs_td["cum_length"].sum(dim=1)  # (batch_size,)
                critic_bl = self.critic(batch)  # (batch_size,)

                actor_loss, critic_loss = self.calculate_loss(log_probs, reward, critic_bl)

                e_rewards = torch.cat([e_rewards, reward.mean().unsqueeze(0)])
                e_lengths = torch.cat([e_lengths, lengths.mean().unsqueeze(0)])
                e_actor_losses = torch.cat([e_actor_losses, actor_loss.unsqueeze(0)])
                e_critic_losses = torch.cat([e_critic_losses, critic_loss.unsqueeze(0)])

                if (epoch == 1 or epoch % self.trainer_params["save_figure_interval"] == 0) and batch_idx == 1:
                    self.env.render(obs_td, actions, save_path=self.figure_dir / f"epoch{epoch}.png")

        return e_rewards.mean(), e_lengths.mean(), e_actor_losses.mean(), e_critic_losses.mean()

    @staticmethod
    def calculate_loss(
        log_prob: torch.Tensor, reward: torch.Tensor, critic_bl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantage = reward - critic_bl  # (batch_size,)

        actor_loss = -torch.mean(advantage.detach() * log_prob.sum(dim=1))
        critic_loss = torch.mean(advantage**2)
        return actor_loss, critic_loss

    def optimize_models(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor):
        for _model, _loss, _optimizer in zip(
            [self.actor, self.critic], [actor_loss, critic_loss], [self.actor_optimizer, self.critic_optimizer]
        ):
            _optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(_model.parameters(), self.trainer_params["max_grad_norm"])  # type: ignore
            _loss.backward()
            _optimizer.step()

    def save_model(self, file_name: str):
        """Save model parameters and optimizer state"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            self.checkpoint_dir / f"{file_name}.pth",
        )

    def log(self, msg: str) -> None:
        self.logger.info(msg)
        self.file_logger.info(msg)


if __name__ == "__main__":
    env_params = {
        "n_custs": 20,
        "n_agents": 2,
        "dimension": 3,
        "min_loc": 0,
        "max_loc": 1,
        "min_demand": 1,
        "max_demand": 1,
        "vehicle_capacity": 1000,
        "one_by_one": False,
        "no_restart": False,
        "imbalance_penalty": True,  # TODO: decay imbalance penalty
        "intermediate_reward": False,  # TODO: support intermediate reward
    }

    model_params = {
        "actor_params": {
            "loc_encoder_params": {"hidden_size": 64},
            "rnn_input_encoder_params": {"hidden_size": 64},
            "ptrnet_params": {"hidden_size": 64, "num_layers": 1, "dropout": 0.05, "glimpse": False},
        },
        "critic_params": {
            "loc_encoder_params": {"hidden_size": 64},
        },
        "actor_optimizer": {"lr": 5e-4},  # TODO: lr scheduler
        "critic_optimizer": {"lr": 5e-4},
    }

    trainer_params = {
        ### CPU or GPU ###
        "device": "cuda",
        ### Training ###
        "n_epochs": 500,
        "early_stop_metric": "reward",  # "reward" or "length"
        "ealry_stop_patience": 20,  # 0 for not using early stopping
        "train_n_samples": 10000,
        "valid_n_samples": 1000,
        "fix_valid_dataset": True,  # use the same validation dataset for all epochs, with seed 0
        "batch_size": 256,
        "max_grad_norm": 2.0,
        ### Logging and Saving ###
        "result_dir": "results",
        "tb_log_dir": "logs",
        "use_tensorboard": True,  # tensorboard --logdir logs
        "save_figure_interval": 10,  # -1 for not saving
        "save_model_interval": 50,  # -1 for not saving
        "exp_name": "debug",
    }

    trainer = MDCVRPTrainer(env_params, model_params, trainer_params)
    trainer.train()
