"""
Trainer for MD-CVRP
"""
from datetime import datetime
from pathlib import Path
from typing import Any
import logging
import sys

from tensorboard_logger import Logger as TbLogger
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from envs import MDCVRPEnv
from models import Actor, Critic, Encoder, PtrNet


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
        self.device = torch.device(self.trainer_params["device"])

        ### Paths ###
        exp_name = f"{env_params['n_nodes']}_{env_params['n_agents']}_{NOW}"
        self.result_dir = Path(self.trainer_params["result_dir"]) / exp_name / "train"
        self.checkpoint_dir = self.result_dir / "checkpoints"
        self.figure_dir = self.result_dir / "figures"
        self.tb_log_dir = None

        for _dir in [self.result_dir, self.checkpoint_dir, self.figure_dir]:
            _dir.mkdir(parents=True, exist_ok=True)

        ### Logger ###
        self.logger, self.tb_logger = logging.getLogger(), None
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.use_tensorboard = self.trainer_params["use_tensorboard"]
        if self.use_tensorboard:
            self.tb_logger = TbLogger(Path(self.trainer_params["tb_log_dir"]) / exp_name)
        self.step_per_epoch = (
            self.trainer_params["train_n_samples"] // self.trainer_params["batch_size"]
            if self.trainer_params["train_n_samples"] % self.trainer_params["batch_size"] == 0
            else self.trainer_params["train_n_samples"] // self.trainer_params["batch_size"] + 1
        )

        ### Env ###
        self.env = MDCVRPEnv(**self.env_params)

        ### Model ###
        loc_encoder = Encoder(**model_params["loc_encoder"]).to(self.device)  # parameter sharing
        self.actor = Actor(
            env=self.env,
            loc_encoder=loc_encoder,
            agent_encoder=Encoder(**model_params["agent_encoder"]),
            rnn_input_encoder=Encoder(**model_params["agent_encoder"]),
            ptrnet=PtrNet(**model_params["ptrnet"]),
            phase="train",
            logger=self.logger,
        ).to(self.device)
        self.critic = Critic(
            env=self.env,
            loc_encoder=Encoder(**model_params["loc_encoder"]),
            hidden_size=model_params["critic_model"]["hidden_size"],
        ).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), **model_params["actor_optimizer"])
        self.critic_optimizer = Adam(self.critic.parameters(), **model_params["critic_optimizer"])

    def train(self) -> None:
        self.logger.info("Start training...")

        n_epochs = self.trainer_params["n_epochs"]

        for epoch in range(n_epochs):
            tr_reward, tr_actor_loss, tr_critic_loss = self.train_epoch(epoch)
            val_reward, val_actor_loss, val_critic_loss = self.validate_epoch(epoch)

            self.logger.info(
                f"Epoch {epoch} TRAIN | "
                + f"Reward: {tr_reward:.3f} | "
                + f"Actor Loss: {tr_actor_loss:.3f} | "
                + f"Critic Loss: {tr_critic_loss:.3f}"
            )
            self.logger.info(
                f"Epoch {epoch} VALID | "
                + f"Reward: {val_reward:.3f} | "
                + f"Actor Loss: {val_actor_loss:.3f} | "
                + f"Critic Loss: {val_critic_loss:.3f}"
            )

            if self.use_tensorboard and self.tb_logger is not None:
                self.tb_logger.log_value("TRAIN reward/epoch", tr_reward, step=epoch)
                self.tb_logger.log_value("TRAIN loss/epoch", tr_actor_loss, step=epoch)
                self.tb_logger.log_value("TRAIN critic_loss/epoch", tr_critic_loss, step=epoch)

                self.tb_logger.log_value("VALID reward/epoch", val_reward, step=epoch)
                self.tb_logger.log_value("VALID loss/epoch", val_actor_loss, step=epoch)
                self.tb_logger.log_value("VALID critic_loss/epoch", val_critic_loss, step=epoch)

            if epoch % self.trainer_params["save_model_interval"] == 0:
                self.save_model(epoch)

        self.logger.info("Finish training...")

    def train_epoch(self, epoch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_dataset = self.env.generate_data(self.trainer_params["train_n_samples"])
        train_dataloader = DataLoader(train_dataset, batch_size=self.trainer_params["batch_size"], collate_fn=lambda x: x)  # type: ignore

        e_rewards = e_actor_losses = e_critic_losses = torch.empty((0,), device=train_dataset.device)
        self.actor.phase = "train"
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            self.env.reset(batch)  # to take steps with mini-batch
            _, _, log_probs, rewards = self.actor(batch)
            rewards = rewards[:, -1]  # (batch_size,), Now, we use only the last reward
            critic_bl = self.critic(batch)  # (batch_size,)

            actor_loss, critic_loss = self.calculate_loss(log_probs, rewards, critic_bl)
            self.optimize_models(actor_loss, critic_loss)

            if self.use_tensorboard and self.tb_logger is not None:
                step = epoch * self.step_per_epoch + batch_idx
                self.tb_logger.log_value("TRAIN reward/step", rewards.mean(), step=step)
                self.tb_logger.log_value("TRAIN actor_loss/step", actor_loss, step=step)
                self.tb_logger.log_value("TRAIN critic_loss/step", critic_loss, step=step)

            e_rewards = torch.cat([e_rewards, rewards.mean().unsqueeze(0)])
            e_actor_losses = torch.cat([e_actor_losses, actor_loss.unsqueeze(0)])
            e_critic_losses = torch.cat([e_critic_losses, critic_loss.unsqueeze(0)])

        return e_rewards.mean(), e_actor_losses.mean(), e_critic_losses.mean()

    def validate_epoch(self, epoch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_dataset = self.env.generate_data(self.trainer_params["valid_n_samples"])
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.trainer_params["batch_size"], collate_fn=lambda x: x)  # type: ignore

        e_rewards = e_actor_losses = e_critic_losses = torch.empty((0,), device=valid_dataset.device)
        with torch.no_grad():
            self.actor.phase = "eval"
            for batch_idx, batch in enumerate(tqdm(valid_dataloader)):
                self.env.reset(batch)  # to take steps with mini-batch
                obs_td, actions, log_probs, rewards = self.actor(batch)
                reward = rewards[:, -1]  # (batch_size,), Now, we use only the last reward
                critic_bl = self.critic(batch)  # (batch_size,)

                actor_loss, critic_loss = self.calculate_loss(log_probs, reward, critic_bl)

                e_rewards = torch.cat([e_rewards, reward.mean().unsqueeze(0)])
                e_actor_losses = torch.cat([e_actor_losses, actor_loss.unsqueeze(0)])
                e_critic_losses = torch.cat([e_critic_losses, critic_loss.unsqueeze(0)])

                if epoch % self.trainer_params["save_figure_interval"] == 0 and batch_idx == 0:
                    self.env.render(batch, actions, save_path=self.figure_dir / f"epoch{epoch}.png")

        return e_rewards.mean(), e_actor_losses.mean(), e_critic_losses.mean()

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

    def save_model(self, epoch: int):
        """Save model parameters and optimizer state"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            self.checkpoint_dir / f"epoch{epoch}.pth",
        )


if __name__ == "__main__":
    import argparse

    from trainer import MDCVRPTrainer

    env_params = {
        "n_nodes": 20,
        "n_agents": 2,
        "min_loc": 0,
        "max_loc": 1,
        "min_demand": 1,
        "max_demand": 9,
        "vehicle_capacity": None,
        "one_by_one": True,
    }

    model_params = {
        "loc_encoder": {"input_size": 3, "hidden_size": 128},  # shared
        "agent_encoder": {"input_size": 3, "hidden_size": 128},  # for actor
        "ptrnet": {"hidden_size": 128, "num_layers": 1, "dropout": 0.05},  # for actor
        "critic_model": {"hidden_size": 128},  # for critic
        "actor_optimizer": {"lr": 5e-4},  # TODO: lr scheduler
        "critic_optimizer": {"lr": 5e-4},
    }

    exp_name = "debug"

    trainer_params = {
        ### Training ###
        "n_epochs": 100,
        "train_n_samples": 10000,
        "valid_n_samples": 1000,
        "batch_size": 256,
        "device": "cuda",
        "max_grad_norm": 1.0,
        ### Logging and Saving ###
        "result_dir": "results",
        "use_tensorboard": True,
        "tb_log_dir": "logs",
        "save_figure_interval": 10,  # -1 for not saving
        "save_model_interval": 10,  # -1 for not saving
        "exp_name": exp_name,
    }

    trainer = MDCVRPTrainer(env_params, model_params, trainer_params)
    trainer.train()
