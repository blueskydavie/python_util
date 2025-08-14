import os
from OpManager import op_manager
from img2vec.img2vec import ModelUtil


# @op_manager.register(name="resnext101_class_5")
class resnext101_class_5(ModelUtil):
    def __init__(self):
        ModelUtil.__init__(
            self,
            model_name="resnext101_32x8d",
            device="cpu",
            num_classes=5,
            pth_path=None
            # pth_path=os.path.join(
            #     os.path.dirname(__file__),
            #     "..",
            #     "..",
            #     "custom-model",
            #     "model",
            #     "fabric-class-5",
            #     "final_epoch",
            #     "model.pth"
            # )
        )

    def get_dim(self):
        return 2048
