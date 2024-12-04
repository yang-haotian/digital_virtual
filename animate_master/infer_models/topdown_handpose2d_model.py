import cv2
import numpy as np
from torch.cuda import nvtx
import torch
from .base_model import BaseModel
from .predictor import numpy_to_torch_dtype_dict
from . import utils


class TopdownHandPose2dModel(BaseModel):
    def __init__(self, **kwargs):
        super(TopdownHandPose2dModel, self).__init__(**kwargs)

    def input_process(self, *data, **kwargs):
        img, bbox = data
        INPUT_H, INPUT_W = self.input_shapes[0][1][1:3]
        input_size = [INPUT_W, INPUT_H]
        c, s = utils._box2cs(bbox, input_size, 1.)
        r = 0
        trans = utils.get_affine_transform(c, s, r, input_size)
        img = cv2.warpAffine(
            img,
            trans, (int(input_size[0]), int(input_size[1])),
            flags=cv2.INTER_LINEAR)[None, ...]
        img = img.astype(np.float32)
        img_info = {"c": c, "r": r, "s": s, "input_size": input_size}
        return img, [img_info]

    def output_process(self, *data, **kwargs):
        pose_preds, img_metas = data
        N = pose_preds.shape[0]
        preds_ret = []
        for i in range(N):
            input_size = img_metas[i]["input_size"]
            H, W = input_size[1] // 4, input_size[0] // 4
            idx = pose_preds[i][:, [0]]
            preds = np.tile(idx, (1, 2)).astype(np.float32)
            preds[:, 0] = preds[:, 0] % W
            preds[:, 1] = preds[:, 1] // W
            scores = pose_preds[i][:, [1]]
            preds = utils.transform_preds(
                preds, img_metas[i]["c"], img_metas[i]["s"], [W, H])
            preds_ret.append(np.concatenate([preds, scores], -1))
        return np.stack(preds_ret)

    def predict_trt(self, *data, **kwargs):
        nvtx.range_push("forward")
        feed_dict = {}
        for i, inp in enumerate(self.predictor.inputs):
            if isinstance(data[i], torch.Tensor):
                feed_dict[inp['name']] = data[i]
            else:
                feed_dict[inp['name']] = torch.from_numpy(data[i]).to(device=self.device,
                                                                      dtype=numpy_to_torch_dtype_dict[inp['dtype']])
        preds_dict = self.predictor.predict(feed_dict, self.cudaStream)
        outs = []
        for i, out in enumerate(self.predictor.outputs):
            output_shape = kwargs.get("output_shape", {})
            if out["name"] in output_shape:
                out_shape = output_shape[out["name"]]
            else:
                out_shape = out["shape"]
            out_tensor = preds_dict[out["name"]][:np.prod(out_shape)].reshape(*out_shape)
            outs.append(out_tensor.cpu().numpy())
        nvtx.range_pop()
        return outs

    def predict(self, *data, **kwargs):
        image, img_info = self.input_process(*data, **kwargs)
        if self.predict_type == "trt":
            preds = self.predict_trt(image, **kwargs)
        else:
            preds = self.predictor.predict(image)
        preds = self.output_process(preds[0], img_info, **kwargs)
        return preds
