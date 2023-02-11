# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from imtt.utils.misc import launch_job
from imtt.utils.parser import load_config, parse_args

# from tools.test_net import test
from tools.test_net_shuffle import test
# from tools.test_net_individuals import test
# from tools.test_net_means import test
from tools.train_net import train as train_full
from tools.train_nocls import train as train_fixcls
# from tools.train_token import train
from tools.train_timeprompt import train
# from tools.train_base_lin import train
# from tools.train_clip import train
# from tools.train_dino import train
# from tools.train_avgtime import train
# from tools.train_baseline import train
# from tools.visualization import visualize
# from timesformer.visualization.tensorboard_vis import TensorboardWriter as visualize


def get_func(cfg):
    train_func = train
    test_func = test
    train_func_full = train_full
    train_func_fixcls = train_fixcls
    return train_func, test_func, train_func_full, train_func_fixcls

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    train, test, train_full, train_fixcls = get_func(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        if cfg.SOLVER.FULL_TUNE_FIXCLS:
            launch_job(cfg=cfg, init_method=args.init_method, func=train_fixcls)
        elif cfg.SOLVER.FULL_TUNE:
            launch_job(cfg=cfg, init_method=args.init_method, func=train_full)
        else:
            launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)


if __name__ == "__main__":
    main()
