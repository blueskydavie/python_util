from OpManager import op_manager
from img2vec.img2vec import ModelUtil

# @op_manager.register(name="resnet101")
class resnet101(ModelUtil):
    def __init__(self):
        ModelUtil.__init__(
                self,
                model_name="resnet101",
                device="cpu",
                num_classes=1000,
                pth_path=None
            )

    def get_dim(self):
        return 2048
