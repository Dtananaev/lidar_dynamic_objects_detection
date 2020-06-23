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

from pylatex import (
    Document,
    Command,
    Section,
    Subsection,
    LongTable,
    MultiColumn,
    Figure,
    SubFigure,
)
from pylatex.utils import italic, bold, NoEscape
import os


def create_long_table(
    doc,
    parameters,
    skip_parameters=[],
    table_specs=r"|p{0.45\linewidth}|p{0.45\linewidth}|",
    header=[bold("Parameter"), bold("Value")],
):
    """
    Helper function to create long table for parameters
    Arguments:
         doc: document to add table
         parameters: parameters dict
         skip_parameters: list of parameters to skip
         table_specs: latex specific table settings
         header: list with column names
    """
    columns = len(header)
    with doc.create(LongTable(table_spec=table_specs)) as data_table:
        # Table header
        data_table.add_hline()
        data_table.add_row(header)
        data_table.add_hline()
        data_table.end_table_header()
        data_table.add_row(
            (MultiColumn(columns, align="r", data="Continued on Next Page"),)
        )
        data_table.end_table_footer()
        data_table.add_row((MultiColumn(columns, align="r", data="End of Table"),))
        data_table.end_table_last_footer()
        for item in parameters:
            if item not in skip_parameters:
                data_table.add_row([item, str(parameters[item])])
                data_table.add_hline()


def add_figure(doc, graphics_dir, image_name, width=r"0.5\linewidth"):
    """
    Helper function to create figure
    Arguments:
        doc: document to add figure
        graphics_dir: directory containing .png image
        image_name: the name of image without extension
        width: width of image in docement page
    """
    image_filename = os.path.join(
        os.path.dirname(__file__), graphics_dir, image_name + ".png"
    )
    with doc.create(Figure(position="h!")) as pic:
        pic.add_image(image_filename, width=NoEscape(width))
        pic.add_caption(image_name)


def add_sub_figure(doc, graphics_dir, image_names=[], captioning="Metrics"):
    """
    Helper function to create multiple sub figures
    Arguments:
        doc: document to add figure
        graphics_dir: directory containing .png image
        image_names: the list of image names without extension
        captioning: global captioning for the figure
    """
    num_figures = len(image_names)
    scale = 1.0 / num_figures
    sub_width = str(scale) + r"\linewidth"

    with doc.create(Figure(position="h!")) as fig:
        for image in image_names:
            image_filename = os.path.join(
                os.path.dirname(__file__), graphics_dir, image + ".png"
            )

            with doc.create(
                SubFigure(position="b", width=NoEscape(sub_width))
            ) as sub_fig:

                sub_fig.add_image(image_filename, width=NoEscape(r"\linewidth"))
                sub_fig.add_caption(image)

        fig.add_caption(captioning)


def generate_latex_pdf(
    graphics_dir,
    output_dir,
    report_dict,
    report_name="experiment_report",
    clean_tex=True,
):
    """
    The function generates  latex/pdf report from json dictionary
    Arguments:
        graphics_dir: directory containing .png images for report
        output_dir: the directory to output report
        report_dict: dictionary with report information
        report_name: the name of output latex/pdf report
        clean_tex: remove latex specific files
    """
    output_filename = os.path.join(output_dir, report_name)

    parameters = report_dict["parameters"]
    report_name = parameters["experiment_info"]["experiment_name"].strip()
    description = parameters["experiment_info"]["description"].strip()
    authors = parameters["experiment_info"]["authors"].strip()

    best_epoch = report_dict["best_epoch"]
    main_metric = report_dict["main_metric"]
    metric_value = float(report_dict["epoch_metrics"][best_epoch][main_metric]) * 100
    result = "\nResult: Best epoch {} with {:.2f}% {}.".format(
        best_epoch, metric_value, main_metric
    )
    # More dertails about page options: https://www.overleaf.com/learn/latex/page_size_and_margins

    geometry_options = {
        "tmargin": "1cm",
        "bmargin": "3cm",
        "lmargin": "2cm",
        "rmargin": "2cm",
        "includeheadfoot": True,
    }
    doc = Document(geometry_options=geometry_options, page_numbers=True)
    doc.preamble.append(Command("title", "Experiment Report"))
    doc.preamble.append(Command("author", authors))
    doc.preamble.append(Command("date", report_dict["date"]))
    doc.append(NoEscape(r"\maketitle"))

    # We should handle in unique way in report each parameter which is not correspod {param : single_value}
    skip_parameters = set(["experiment_info", "optimizer", "scheduler", "augment"])

    with doc.create(Section(report_name)):
        doc.append(italic("Description:\n"))
        doc.append(description)
        doc.append(bold(result))

        with doc.create(Subsection("Parameters")):
            create_long_table(doc, parameters, skip_parameters)

            with doc.create(Subsection("Optimizer")):
                create_long_table(doc, parameters["optimizer"])
            with doc.create(Subsection("Scheduler")):
                create_long_table(doc, parameters["scheduler"])
                add_figure(doc, graphics_dir, "learning_rate_scheduler")

            with doc.create(Subsection("Augmentations")):
                create_long_table(doc, parameters["augment"])

    with doc.create(Section("Data plots")):
        image_names = ["loss_epoch_metrics"]
        add_sub_figure(
            doc, graphics_dir, image_names=image_names, captioning="Epoch metrics"
        )
        # add_figure(doc, graphics_dir, "accuracy_epoch_metrics")

    doc.generate_pdf(output_filename, clean_tex=clean_tex)
