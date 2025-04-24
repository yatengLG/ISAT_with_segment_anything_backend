# -*- coding: utf-8 -*-
# @Author  : LG

from ISAT_SAM_BACKEND.segment_any.segment_any import SegAny
import numpy as np
import torch


class SAM:
    _instance = None

    def __init__(self):
        self.segany:SegAny = None

    def load_model(self, model_path, use_bfloat16=False):
        if not torch.cuda.is_available():
            use_bfloat16 = False

        self.segany = SegAny(model_path, use_bfloat16=use_bfloat16)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def inference(self, image: np.ndarray):
        with torch.inference_mode(), torch.autocast(self.segany.device,
                                                    dtype=self.segany.model_dtype,
                                                    enabled=torch.cuda.is_available()):
            if 'sam2' in self.segany.model_type:
                _orig_hw = tuple([image.shape[:2]])
                input_image = self.segany.predictor_with_point_prompt._transforms(image)
                input_image = input_image[None, ...].to(self.segany.predictor_with_point_prompt.device)
                backbone_out = self.segany.predictor_with_point_prompt.model.forward_image(input_image)
                _, vision_feats, _, _ = self.segany.predictor_with_point_prompt.model._prepare_backbone_features(
                    backbone_out)
                if self.segany.predictor_with_point_prompt.model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[
                                           -1] + self.segany.predictor_with_point_prompt.model.no_mem_embed
                feats = [
                            feat.permute(1, 2, 0).view(1, -1, *feat_size)
                            for feat, feat_size in
                            zip(vision_feats[::-1], self.segany.predictor_with_point_prompt._bb_feat_sizes[::-1])
                        ][::-1]
                _features = {"image_embed": feats[-1], "high_res_feats": tuple(feats[:-1])}
                return _features, _orig_hw, _orig_hw
            else:
                input_image = self.segany.predictor_with_point_prompt.transform.apply_image(image)
                input_image_torch = torch.as_tensor(input_image,
                                                    device=self.segany.predictor_with_point_prompt.device)
                input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

                original_size = image.shape[:2]
                input_size = tuple(input_image_torch.shape[-2:])

                input_image = self.segany.predictor_with_point_prompt.model.preprocess(input_image_torch)
                features = self.segany.predictor_with_point_prompt.model.image_encoder(input_image)
            return features, original_size, input_size

    def postprocess(self, features, original_size, input_size):
        if self.segany.model_source == 'sam_hq':
            # sam_hq features is a tuple. include features: Tensor and interm_features: List[Tensor, ...]
            features, interm_features = features

            features = features.detach().to(torch.float32).cpu().numpy().tolist()
            interm_features = [interm_feature.detach().to(torch.float32).cpu().numpy().tolist() for interm_feature in interm_features]
            features = (features, interm_features)

        elif 'sam2' in self.segany.model_source:
            # sam2 features is a dict. include image_embed: Tensor and high_res_feats: Tuple[Tensor, ...]
            image_embed = features['image_embed']
            high_res_feats = features['high_res_feats']

            image_embed = image_embed.detach().to(torch.float32).cpu().numpy().tolist()
            high_res_feats = [high_res_feat.detach().to(torch.float32).cpu().numpy().tolist() for high_res_feat in high_res_feats]
            features['image_embed'] = image_embed
            features['high_res_feats'] = high_res_feats
        else:
            features = features.detach().cpu().numpy().tolist()
        return features, original_size, input_size

    def predict(self, image: np.ndarray):
        if self.segany is None:
            raise RuntimeError('SAM model has not been initialized')
        features, original_size, input_size = self.inference(image)
        features, original_size, input_size = self.postprocess(features, original_size, input_size)
        return features, original_size, input_size


# 全局调用模型
sam = SAM.get_instance()
