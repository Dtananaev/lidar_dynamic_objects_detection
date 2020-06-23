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

import argparse
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from detection_3d.tools.file_io import save_to_json, save_plot_to_image, read_json
from detection_3d.tools.report_tools import plot_metric_to_image, get_report_json

from detection_3d.parameters import SchedulerSettings
from detection_3d.tools.training_helpers import get_optimizer
from detection_3d.tools.pylatex_tools import generate_latex_pdf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Ensure report runs on cpu


def plot_learning_rate_to_file(graphics_dir, report_dict):
    """
    The function plots learning rate graph to png image
    Arguments:
        graphics_dir: directory to save images
        report_dict: dictionary containing learning rate scheduler settings
    """
    best_epoch = int(report_dict["best_epoch"])
    main_metric = report_dict["main_metric"]
    filename_to_save = os.path.join(graphics_dir, "learning_rate_scheduler.png")

    current_epoch = int(report_dict["plot_metrics"][main_metric]["epoch"][-1])
    steps_per_epoch = int(
        report_dict["parameters"]["train_size"]
        / report_dict["parameters"]["batch_size"]
    )

    total_iterations = steps_per_epoch * current_epoch
    steps = np.arange(0, total_iterations).astype(
        float
    )  # triangular_scheduler learning rate needs float dtype
    if report_dict["parameters"]["scheduler"] == SchedulerSettings.no_scheduler():
        lr = (
            np.ones(total_iterations)
            * report_dict["parameters"]["scheduler"]["initial_learning_rate"]
        )
    else:
        lr, _ = get_optimizer(
            report_dict["parameters"]["optimizer"],
            report_dict["parameters"]["scheduler"],
            steps_per_epoch,
        )
        lr = lr(steps)
    metrics_dict = {"learning_rate": {"epoch": steps / steps_per_epoch, "value": lr}}
    plot_metric_to_image(
        metrics_dict,
        filename_to_save,
        plot_title=report_dict["parameters"]["scheduler"]["name"],
        best_epoch=best_epoch,
    )


def plot_metric_to_file(graphics_dir, metrics_dict, metric_name, best_epoch=None):
    """
    The function plots specified metric graph to png image
    Arguments:
        graphics_dir: directory to save images
        metrics_dict: dictionary containing metrics over different epochs
        metric_name: metric name to plot
        best_epoch: plot dotted red line if best epoch is given
    """
    subdict = {
        metric: metrics_dict[metric] for metric in metrics_dict if metric_name in metric
    }
    filename_to_save = os.path.join(graphics_dir, metric_name + "_epoch_metrics.png")
    plot_metric_to_image(subdict, filename_to_save, best_epoch=best_epoch)


def report(
    checkpoints_dir,
    report_dir,
    from_file=None,
    full_report=False,
    latex_pdf=True,
    remove_plots=True,
):
    """
    The generate json report and save it and also generate pdf report with plots
    Arguments:
        checkpoints_dir: ditectory containing all checkpoints from training
        report_dir: directory to save reprot
        from_file: if not None generate pdf report from specified .json file
        full_report: generate .json file containing all information
                     from all epoch (not only from the best)
        latex_pdf: if true generate pdf file otherwise only .json
        remove_plots: if true removes all generated .png plots
    """
    if from_file is None:
        report_json = os.path.join(report_dir, "experiment_report.json")
        graphics_dir = os.path.join(report_dir, "experiment_report")
        os.makedirs(graphics_dir, exist_ok=True)

        report_name = "experiment_report"
        # Metrics to plot
        plot_metrics = ["train_loss", "val_loss"]
        main_metric = "val_loss"  # Metric to define best epoch
        report_dict = get_report_json(
            checkpoints_dir,
            report_dir,
            plot_metrics,
            main_metric,
            full_report=full_report,
        )
        save_to_json(report_json, report_dict)

    else:
        report_dict = read_json(from_file)
        graphics_dir = os.path.splitext(from_file)[0]
        report_dir = "/".join(graphics_dir.split("/")[:-1])
        report_name = graphics_dir.split("/")[-1]
        os.makedirs(graphics_dir, exist_ok=True)

    # Get metrics to plot
    plot_dict = report_dict["plot_metrics"]
    best_epoch = int(report_dict["best_epoch"])

    # Create images of plots for: loss, miou, accuracy
    plot_metric_to_file(
        graphics_dir, plot_dict, metric_name="loss", best_epoch=best_epoch
    )
    plot_learning_rate_to_file(graphics_dir, report_dict)

    if latex_pdf:
        generate_latex_pdf(graphics_dir, report_dir, report_dict, report_name)
    if remove_plots:
        shutil.rmtree(graphics_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create report.")
    parser.add_argument(
        "--full",
        type=lambda x: x,
        nargs="?",
        const=True,
        default=False,
        help="Generate full report with raw information over all epochs (not only best epoch).",
    )
    parser.add_argument("--checkpoints_dir", default="log/checkpoints")
    parser.add_argument("--report_dir", default="log/report")
    parser.add_argument("--from_file", type=str, default=None)
    args = parser.parse_args()
    report(args.checkpoints_dir, args.report_dir, args.from_file, args.full)
