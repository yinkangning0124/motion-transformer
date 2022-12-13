import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):
    def train_step(self, env):
        (
            states,
            dones,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)
        state_target = torch.clone(states)
        
        state_preds= self.model.forward(
            states,
            timesteps,
            attention_mask=attention_mask,
        )

        state_target = state_target[ : , 1 : , : ]
        state_preds = state_preds[ : , 1 : , : ]
        attention_mask = attention_mask[:, 1 : ]
        
        state_dim = state_preds.shape[2]
        
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        '''
        rotation_target = state_target[:, : 96]
        root_target = state_target[:, 96 : 99]

        rotation_pred = state_preds[:, : 96]
        root_pred = state_preds[:, 96 : 99]

        rotation_target = rotation_target.reshape(self.batch_size * 24, 4)
        rotation_pred = rotation_pred.reshape(self.batch_size * 24, 4)

        loss_rotation = self.quat_diff_rad(rotation_pred, rotation_target) # [batchsize * 24]
        loss_rotation = loss_rotation.mean()

        loss_root = self.loss_fn(
            root_target,
            None,
            None,
            root_pred,
            None,
            None,
        )
        loss = loss_rotation + loss_root
        '''
        
        '''display state_transition in mujoco using set_state()
        
        qpos = state_target[0, 0:5].detach().cpu().numpy()
        qpos = np.concatenate(
            (
                np.zeros(1),
                qpos,
            )
        )
        qvel = state_target[0, 5:11].detach().cpu().numpy()
        env.set_state(qpos, qvel)
        env.render()
        '''
        
        
        loss = self.loss_fn(
            state_preds,
            None,
            None,
            state_target,
            None,
            None,
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((state_preds - state_target) ** 2).detach().cpu().item()
            )
        
        return loss.detach().cpu().item()


    def quat_conjugate(self, a):
        shape = a.shape
        a = a.reshape(-1, 4)
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    
    def quat_mul(self, a, b):
        assert a.shape == b.shape
        shape = a.shape
        a = a.reshape(-1, 4)
        b = b.reshape(-1, 4)

        x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)

        quat = torch.stack([x, y, z, w], dim=-1).view(shape)

        return quat

    def quat_diff_rad(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Get the difference in radians between two quaternions.

        Args:
            a: first quaternion, shape (N, 4)
            b: second quaternion, shape (N, 4)
        Returns:
            Difference in radians, shape (N,)
        """
        b_conj = self.quat_conjugate(b)
        mul = self.quat_mul(a, b_conj)
        # 2 * torch.acos(torch.abs(mul[:, -1]))
        return 2.0 * torch.asin(
            torch.clamp(
                torch.norm(
                    mul[:, 0:3],
                    p=2, dim=-1), max=1.0)
    )