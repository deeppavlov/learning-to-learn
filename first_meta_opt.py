import tensorflow as tf
from meta import Meta


class FirstMetaOpt(Meta):
    def __init__(self,
                 num_exercises=10,
                 os_num_nodes=256,
                 unr_num_nodes=256,
                 num_f=10):