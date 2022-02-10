import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from pytorch_trainer.iterators import MultiprocessIterator
from pytorch_trainer.training import Trainer, extensions
from pytorch_trainer.training.updaters import StandardUpdater
from tensorboardX import SummaryWriter

from yukarin_saa.config import Config
from yukarin_saa.dataset import create_dataset
from yukarin_saa.evaluator import GenerateEvaluator
from yukarin_saa.generator import Generator
from yukarin_saa.model import Model
from yukarin_saa.network.predictor import create_predictor
from yukarin_saa.utility.pytorch_utility import AmpUpdater, init_weights, make_optimizer
from yukarin_saa.utility.trainer_extension import (
    NoamShift,
    TensorboardReport,
    WandbReport,
)
from yukarin_saa.utility.trainer_utility import (
    LowValueTrigger,
    create_iterator,
    list_concat,
)


def create_trainer(
    config_dict: Dict[str, Any],
    output: Path,
):
    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()

    output.mkdir(exist_ok=True, parents=True)
    with (output / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # model
    predictor = create_predictor(config.network)
    model = Model(model_config=config.model, predictor=predictor)
    if config.train.weight_initializer is not None:
        init_weights(model, name=config.train.weight_initializer)

    device = torch.device("cuda") if config.train.use_gpu else torch.device("cpu")
    model.to(device)

    # dataset
    _create_iterator = partial(
        create_iterator,
        batch_size=config.train.batch_size,
        num_processes=config.train.num_processes,
        use_multithread=config.train.use_multithread,
    )

    datasets = create_dataset(config.dataset)
    train_iter = _create_iterator(datasets["train"], for_train=True)
    test_iter = _create_iterator(datasets["test"], for_train=False)
    eval_iter = _create_iterator(datasets["test"], for_train=False, for_eval=True)

    valid_iter = None
    if datasets["valid"] is not None:
        valid_iter = _create_iterator(datasets["valid"], for_train=False, for_eval=True)

    warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)

    # optimizer
    optimizer = make_optimizer(config_dict=config.train.optimizer, model=model)

    # updater
    if not config.train.use_amp:
        updater = StandardUpdater(
            iterator=train_iter,
            optimizer=optimizer,
            model=model,
            converter=list_concat,
            device=device,
        )
    else:
        updater = AmpUpdater(
            iterator=train_iter,
            optimizer=optimizer,
            model=model,
            converter=list_concat,
            device=device,
        )

    # trainer
    trigger_log = (config.train.log_iteration, "iteration")
    trigger_eval = (config.train.eval_iteration, "iteration")
    trigger_snapshot = (config.train.snapshot_iteration, "iteration")
    trigger_stop = (
        (config.train.stop_iteration, "iteration")
        if config.train.stop_iteration is not None
        else None
    )

    trainer = Trainer(updater, stop_trigger=trigger_stop, out=output)

    if (
        config.train.step_shift is not None
        and config.train.step_shift["step"] is not None
    ):
        ext = extensions.StepShift(**config.train.step_shift)
        trainer.extend(ext)

    if (
        config.train.noam_shift is not None
        and config.train.noam_shift["step"] is not None
    ):
        ext = NoamShift(**config.train.noam_shift)
        trainer.extend(ext)

    ext = extensions.Evaluator(test_iter, model, converter=list_concat, device=device)
    trainer.extend(ext, name="test", trigger=trigger_log)

    generator = Generator(
        config=config, predictor=predictor, use_gpu=config.train.use_gpu
    )
    generate_evaluator = GenerateEvaluator(generator=generator)
    ext = extensions.Evaluator(
        eval_iter, generate_evaluator, converter=list_concat, device=device
    )
    trainer.extend(ext, name="eval", trigger=trigger_eval)
    if valid_iter is not None:
        ext = extensions.Evaluator(
            valid_iter, generate_evaluator, converter=list_concat, device=device
        )
        trainer.extend(ext, name="valid", trigger=trigger_eval)

    if config.train.stop_iteration is not None:
        saving_model_num = int(
            config.train.stop_iteration / config.train.snapshot_iteration / 10
        )
    else:
        saving_model_num = 10
    ext = extensions.snapshot_object(
        predictor,
        filename="predictor_{.updater.iteration}.pth",
        n_retains=saving_model_num,
    )
    trainer.extend(
        ext,
        trigger=LowValueTrigger("test/main/loss", trigger=trigger_eval),
    )

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.observe_lr(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(
        extensions.PrintReport(["iteration", "main/loss", "test/main/loss"]),
        trigger=trigger_log,
    )

    ext = TensorboardReport(writer=SummaryWriter(Path(output)))
    trainer.extend(ext, trigger=trigger_log)

    if config.project.category is not None:
        ext = WandbReport(
            config_dict=config.to_dict(),
            project_category=config.project.category,
            project_name=config.project.name,
            output_dir=output.joinpath("wandb"),
        )
        trainer.extend(ext, trigger=trigger_log)

    (output / "struct.txt").write_text(repr(model))

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    ext = extensions.snapshot_object(
        trainer,
        filename="trainer_{.updater.iteration}.pth",
        n_retains=1,
        autoload=True,
    )
    trainer.extend(ext, trigger=trigger_snapshot)

    return trainer
