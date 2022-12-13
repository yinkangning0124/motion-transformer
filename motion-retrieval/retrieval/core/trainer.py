import torch

from retrieval.logger import logger
from retrieval.utils import pytorch_util as ptu
from retrieval.utils.pytorch_util import PyTorchModule


class Trainer:
    def __init__(
        self,
        embed_model: PyTorchModule,
        pca_model,
        pred_future_steps: int = 5,
        embed_lr: float = 1e-3,
        num_embed_epoches: int = 10,
        condition_on_target: bool = False,
        train_dataloader: torch.utils.data.DataLoader = None,
        test_dataloader: torch.utils.data.DataLoader = None,
    ):
        self.embed_model = embed_model
        self.pca_model = pca_model
        self.num_embed_epoches = num_embed_epoches
        self.pred_future_steps = pred_future_steps
        self.condition_on_target = condition_on_target

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.embed_loss = torch.nn.MSELoss()
        self.embed_optimizer = torch.optim.Adam(
            self.embed_model.parameters(), lr=embed_lr
        )
        self.best_loss = float("inf")

    def train(self):
        assert self.embed_model is not None
        self.train_embed_model()
        if self.pca_model is not None:
            self.train_pca_model()

    def train_embed_model(self):
        for epoch in range(self.num_embed_epoches):
            logger.push_prefix("Iteration #%d | " % epoch)
            self.eval_embed_model(epoch)
            self._update_embed_model_one_epoch()
            logger.pop_prefix()

    def eval_embed_model(self, epoch: int):
        logger.log("Evaluate on the test set")
        loss = 0.
        total_cnt = 0

        for batch in self.test_dataloader:
            obs, future_obs = batch
            obs, future_obs = obs.to(ptu.device), future_obs.to(ptu.device)
            future_obs = future_obs[:, :self.pred_future_steps]

            if self.condition_on_target:
                initial_input = torch.cat([obs, future_obs[:, self.pred_future_steps - 1]], dim=1)
            else:
                initial_input = obs
            next_obs, encode_state = self.embed_model.initial_inference(initial_input)
            pred_future_obs = [next_obs]
            for _ in range(1, self.pred_future_steps):
                next_obs, encode_state = self.embed_model.recurrent_inference(encode_state)
                pred_future_obs.append(next_obs)
            pred_future_obs = torch.stack(pred_future_obs, dim=1)

            batch_loss = self.embed_loss(pred_future_obs, future_obs)
            batch_cnt = obs.shape[0]
            delta = batch_loss - loss
            total_cnt = total_cnt + batch_cnt
            loss = loss + delta * batch_cnt / total_cnt

        if loss < self.best_loss:
            self.best_loss = loss
            data_to_save = {"Eval Embed Loss": loss.item()}
            data_to_save.update(self.get_epoch_snapshot(epoch))
            logger.save_extra_data(data_to_save, "best.pkl")
        logger.record_tabular("Eval Embed Loss", loss.item())
        logger.record_tabular("Epoch", epoch)

    def train_pca_model(self):
        pass

    def get_epoch_snapshot(self, epoch: int):
        data_to_save = dict(
            epoch=epoch,
            embed_model=self.embed_model,
        )
        return data_to_save

    def _update_embed_model_one_epoch(self):
        loss = 0.
        total_cnt = 0

        for batch in self.train_dataloader:
            obs, future_obs = batch
            obs, future_obs = obs.to(ptu.device), future_obs.to(ptu.device)
            future_obs = future_obs[:, :self.pred_future_steps]

            if self.condition_on_target:
                initial_input = torch.cat([obs, future_obs[:, self.pred_future_steps - 1]], dim=1)
            else:
                initial_input = obs
            next_obs, encode_state = self.embed_model.initial_inference(initial_input)
            pred_future_obs = [next_obs]
            for _ in range(1, self.pred_future_steps):
                next_obs, encode_state = self.embed_model.recurrent_inference(encode_state)
                pred_future_obs.append(next_obs)
            pred_future_obs = torch.stack(pred_future_obs, dim=1)

            batch_loss = self.embed_loss(pred_future_obs, future_obs)
            self.embed_optimizer.zero_grad()
            batch_loss.backward()
            self.embed_optimizer.step()

            batch_cnt = obs.shape[0]
            delta = batch_loss - loss
            total_cnt = total_cnt + batch_cnt
            loss = loss + delta * batch_cnt / total_cnt

        logger.record_tabular("Train Embed Loss", loss.item())
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def to(self, device):
        self.embed_model.to(device)
