from OpManager import op_manager
from img2vec.img2vec import ModelUtil

# @op_manager.register(name="resnext101")
class resnext101(ModelUtil):
    def __init__(self):
        ModelUtil.__init__(
                self,
                model_name="resnext101_32x8d",
                device="cpu",
                num_classes=1000,
                pth_path=None
                )

    def get_dim(self):
        return 2048
