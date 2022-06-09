import os
from PIL import Image
from pycallgraph2 import PyCallGraph, Config, GlobbingFilter
from pycallgraph2.output import GraphvizOutput

from lung import create_app
from lung.core.analyzer import analyze_one, train_all
from test import example_analysis_show

im = Image.open(os.path.join(os.path.dirname(__file__), "lung/data/images/3-100-2.jpg"))
app = create_app()
config = Config()
config.trace_filter = GlobbingFilter(
    include=['lung.*'], exclude=['lung.core.simclr.*','lung.core.yolov5.*']
)

class MyOutput(GraphvizOutput):

    def node_label(self, node):
        my_name = "\n->".join(node.name.split(".")[2:])
        # node.name = my_name

        parts = [
            my_name,
            # '\n',
            # 'calls: {0.calls.value:n}',
            # 'time: {0.time.value:f}s',
        ]

        # if self.processor.config.memory:
        #     parts += [
        #         'memory in: {0.memory_in.value_human_bibyte}',
        #         'memory out: {0.memory_out.value_human_bibyte}',
        #     ]

        return r'\n'.join(parts).format(node)
        # return my_name

graphviz = MyOutput(output_file='giving_scores.png', font_size = 20, group_font_size = 30)

with app.app_context():
    analyze_one(im)
    with PyCallGraph(output=graphviz, config=config):
        analyze_one(im)
        # train_all(simclr_epoch=0, yolo_epoch=0, test_ratio=1)
        # example_analysis_show()

####################################################################################################
# 
####################################################################################################

config = Config()
config.trace_filter = GlobbingFilter(
    include=['lung.*'], exclude=['lung.core.simclr.*','lung.core.yolov5.*']
)

class MyOutput2(GraphvizOutput):

    def node_label(self, node):
        my_name = "\n->".join(node.name.split(".")[2:])
        # node.name = my_name

        parts = [
            my_name,
            # '\n',
            # 'calls: {0.calls.value:n}',
            # 'time: {0.time.value:f}s',
        ]

        # if self.processor.config.memory:
        #     parts += [
        #         'memory in: {0.memory_in.value_human_bibyte}',
        #         'memory out: {0.memory_out.value_human_bibyte}',
        #     ]

        return r'\n'.join(parts).format(node)

graphviz = MyOutput2(output_file='training.png', font_size = 20, group_font_size = 30)

with app.app_context():
    with PyCallGraph(output=graphviz, config=config):
        train_all(simclr_epoch=0, yolo_epoch=0, test_ratio=1)