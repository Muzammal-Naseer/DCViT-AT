# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

from . import logging
from . import metrics

logger = logging.get_logger(__name__)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
        depth=1,
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        self.depth = depth
        # Initialize tensors.
        # self.video_preds = torch.zeros((num_videos, num_cls))
        self.video_preds = {}
        for i in range(self.depth):
            self.video_preds[i] = torch.zeros((num_videos, num_cls))

        if multi_label:
            for i in range(self.depth):
                self.video_preds[i] -= 1e10

        self.video_labels = (
            torch.zeros((num_videos, num_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        for layer in self.video_preds:
            self.video_preds[layer].zero_()
        if self.multi_label:
            for layer in self.video_preds:
                self.video_preds[layer] -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        # print(len(preds), preds[0].shape)
        for num_layer, pred in enumerate(preds):
            for ind in range(pred.shape[0]):
                vid_id = int(clip_ids[ind]) // self.num_clips
                if self.video_labels[vid_id].sum() > 0:
                    assert torch.equal(
                        self.video_labels[vid_id].type(torch.FloatTensor),
                        labels[ind].type(torch.FloatTensor),
                    )
                self.video_labels[vid_id] = labels[ind]
                if self.ensemble_method == "sum":
                    self.video_preds[num_layer][vid_id] += pred[ind]
                elif self.ensemble_method == "max":
                    self.video_preds[num_layer][vid_id] = torch.max(
                        self.video_preds[num_layer][vid_id], pred[ind]
                    )
                else:
                    raise NotImplementedError(
                        "Ensemble Method {} is not supported".format(
                            self.ensemble_method
                        )
                    )
                if num_layer == 0:
                    self.clip_count[vid_id] += 1

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        for i in range(self.depth):
            num_topks_correct = metrics.topks_correct(
                self.video_preds[i], self.video_labels, ks
            )
            topks = [
                (x / self.video_preds[i].size(0)) * 100.0
                for x in num_topks_correct
            ]

            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                self.stats["layer_{}: top{}_acc".format(i+1,k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        return self.stats

class CompareMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        ensemble_method="sum",
        depth=1,
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method
        self.depth = depth
        # Initialize tensors.
        # self.video_preds = torch.zeros((num_videos, num_cls))
        self.video_preds = {}
        for i in range(self.depth):
            self.video_preds[i] = torch.zeros((num_videos, num_cls))

        self.video_labels = {}
        for i in range(self.depth):
            self.video_labels[i] = torch.zeros((num_videos, num_cls))

        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        for layer in self.video_preds:
            self.video_preds[layer].zero_()
            
        for layer in self.video_labels:
            self.video_labels[layer].zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for num_layer, pred in enumerate(preds):
            for ind in range(pred.shape[0]):
                vid_id = int(clip_ids[ind]) // self.num_clips

                if self.ensemble_method == "sum":
                    self.video_preds[num_layer][vid_id] += pred[ind]
                    self.video_labels[num_layer][vid_id] += labels[num_layer][ind]
                elif self.ensemble_method == "max":
                    self.video_preds[num_layer][vid_id] = torch.max(
                        self.video_preds[num_layer][vid_id], pred[ind]
                    )
                    self.video_labels[num_layer][vid_id] = torch.max(
                        self.video_labels[num_layer][vid_id], labels[num_layer][ind]
                    )
                else:
                    raise NotImplementedError(
                        "Ensemble Method {} is not supported".format(
                            self.ensemble_method
                        )
                    )
                if num_layer == 0:
                    self.clip_count[vid_id] += 1

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        for i in range(self.depth):
            fooled = torch.sum(self.video_preds[i].argmax(dim=-1) != self.video_labels[i].argmax(dim=-1)).item()
            self.stats["layer_{}: fooled".format(i+1)] = (fooled / self.video_preds[i].shape[0]) * 100.0
        return self.stats

