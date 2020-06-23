#!/usr/bin/env python
__copyright__ = """
Copyright (c) 2020 Tananaev Denis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from detection_3d.tools.file_io import read_json, save_plot_to_image


def parse_report_for_epoch_metrics(report_dict, metrics=["train_mIoU"]):
    """
    Parse report for given metric information over epoch and add them to list
    Arguments:
        report_dict: dictionalry with raw report data
        metrics: list of metrics to search over epoch
    Returns:
        result: the dictionary of the form  {metric : {"epoch": [], "value": []}
    """

    result = {metric: {"epoch": [], "value": []} for metric in metrics}
    for epoch in report_dict["epoch_metrics"]:
        epoch_num = int(epoch)
        for metric in metrics:
            value = float(report_dict["epoch_metrics"][epoch][metric])
            result[metric]["epoch"].append(epoch_num)
            result[metric]["value"].append(value)

    return result


def get_report_json(
    checkpoints_dir, report_dir, plot_metrics, main_metric, full_report=False
):
    """
    Parse checkpoints_dir and returns json report with all information
    Arguments:
        checkpoints_dir: ditectory containing all checkpoints from training
        report_dir: directory to save reprot
        plot_metrics: metrics to plot
        main_metric: main metric to define best epoch
        full_report: generate .json file containing all information
                     from all epoch (not only from the best)
    Returns:
        report_dict: the dictionary with information for report
    """
    search_string = os.path.join(checkpoints_dir, "*")
    model_list = sorted(glob.glob(search_string))
    date_now = datetime.datetime.now().strftime("%d %B %Y")

    report_dict = {
        "model_name": None,
        "date": date_now,
        "parameters": None,
        "best_epoch": None,
        "main_metric": main_metric,
        "epoch_metrics": None,
        "plot_metrics": None,
    }

    if len(model_list) == 0:
        ValueError("The checkpoint folder {} is empty".format(checkpoints_dir))
    else:
        epoch_metrics = {}
        for model_folder in model_list:
            model_name, epoch = model_folder.split("/")[-1].split("-")

            # Fill header
            if report_dict["model_name"] is None and report_dict["parameters"] is None:
                param_filename = os.path.join(model_folder, "parameters.json")
                report_dict["model_name"] = model_name
                report_dict["parameters"] = read_json(param_filename)
            # Check that we have only one model name inside checkpoints
            if model_name != report_dict["model_name"]:
                ValueError(
                    "model name in report {} is not \
                            same as current model folder {}".format(
                        report_dict["model_name"], model_name
                    )
                )
            # Fill epoch metrics
            metrics_filename = os.path.join(model_folder, "epoch_metrics.json")
            epoch_metrics[epoch] = read_json(metrics_filename)
        # Find best epoch by metric

        report_dict["epoch_metrics"] = epoch_metrics

        # Get metrics to plot
        plot_metrics = parse_report_for_epoch_metrics(report_dict, metrics=plot_metrics)

        # Find best checkpoint idx, epoch
        best_ckpt_idx = np.argmax(plot_metrics[main_metric]["value"])
        best_epoch = "{0:04d}".format(plot_metrics[main_metric]["epoch"][best_ckpt_idx])
        report_dict["best_epoch"] = best_epoch
        report_dict["plot_metrics"] = plot_metrics
        if not full_report:
            # Override the epoch metrics with info from only best epoch
            report_dict["epoch_metrics"] = {best_epoch: epoch_metrics[best_epoch]}

    return report_dict


def plot_metric_to_image(
    metrics_dict,
    filename_to_save,
    plot_title="epoch_metrics",
    loc="best",
    best_epoch=None,
):

    """
    Plot metrics over multiple epoch (e.g. train/val loss) as graph to image
    Arguments:
          metrics_dict: dict of the type metric : {"epoch": [], "value": []}
          filename_to_save: image filename to save plot
          plot_title: title of plot
          loc: location of the legend (see matplotlib options)
          best_epoch: plot dotted line for best epoch
    """
    plt.style.use("seaborn-whitegrid")
    figure = plt.figure()
    plt.title(plot_title)

    plt.xlabel("epoch")
    plt.ylabel("value")
    if best_epoch is not None:
        plt.axvline(
            best_epoch,
            0,
            1,
            linestyle=":",
            color="tab:red",
            alpha=1.0,
            label="best checkpoint",
        )
    for metric in metrics_dict:
        color = "tab:blue" if "val" in metric else "tab:orange"

        x = metrics_dict[metric]["epoch"]
        y = metrics_dict[metric]["value"]
        plt.plot(x, y, label=metric, color=color, linewidth=1, alpha=1.0)
        plt.legend(loc=loc)
    save_plot_to_image(filename_to_save, figure)
    plt.style.use("classic")  # Change back to classic style
