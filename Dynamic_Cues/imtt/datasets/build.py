# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split, si=0):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    name = dataset_name.capitalize()
    if name.lower() == 'multiviewimgdataset':
        pth = cfg.DATA.PATH_TO_DATA_DIR
        if split == 'val' or split == 'test':
            pth += 'test'
        else: 
            pth += 'train'
        return DATASET_REGISTRY.get(name)(pth, test_mode=split=='test')
        # return DATASET_REGISTRY.get(name)(pth)
    try:
        return DATASET_REGISTRY.get(name)(cfg, split, si=si)
    except:
        return DATASET_REGISTRY.get(name)(cfg, split)
