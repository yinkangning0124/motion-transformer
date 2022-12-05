import yaml
import argparse

from retrieval.utils import pytorch_util as ptu
from retrieval.utils.dataset import MotionDataLoader
from retrieval.launchers.launcher_util import setup_logger, set_seed
from retrieval.core.models import MuZeroFullyConnected
from retrieval.core.trainer import Trainer


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)

    train_file_path = listings[variant["expert_name"]]["train_file_path"][0]
    test_file_path = listings[variant["expert_name"]]["test_file_path"][0]

    train_dataloader = MotionDataLoader(data_path=train_file_path, batch_size=variant["batch_size"], shuffle=True)
    test_dataloader = MotionDataLoader(data_path=test_file_path, batch_size=variant["batch_size"])

    muzero_params = variant['muzero_params']
    model = MuZeroFullyConnected(
        obs_dim=train_dataloader.obs_dim,
        condition_on_target=variant["condition_on_target"],
        **muzero_params,
    )

    trainer = Trainer(
        embed_model=model,
        pca_model=None,  # not implemented yet
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        condition_on_target=variant["condition_on_target"],
        **variant['trainer_params'],
    )

    if ptu.gpu_enabled():
        trainer.to(ptu.device)
    trainer.train()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    if exp_specs["using_gpus"]:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs["exp_id"]

    exp_prefix = exp_specs["exp_name"]
    exp_prefix += "{}-embedlr_{}-embeddim_{}-fstep_{}".format(
        "" if not exp_specs["condition_on_target"] else "-cond",
        exp_specs["trainer_params"]["embed_lr"],
        exp_specs["muzero_params"]["embed_dim"],
        exp_specs["trainer_params"]["pred_future_steps"],
    )

    seed = exp_specs["seed"]
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    experiment(exp_specs)
