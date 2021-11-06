import json
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

if __name__ == '__main__':
    convert_tf_checkpoint_to_pytorch("../../data/pretrained_models/model-ernie1.0.12/model.ckpt",
                                     "../../data/pretrained_models/model-ernie1.0.1/labert_config.json",
                                     "../../data/pretrained_models/model-ernie1.0.1/pytorch_model.bin")
