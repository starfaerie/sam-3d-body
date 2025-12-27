# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Optional, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        # NEW: Keypoint conditioning parameters
        keypoints_2d: Optional[np.ndarray] = None,
        keypoints_3d: Optional[np.ndarray] = None,
        keypoint_scores: Optional[np.ndarray] = None,
        keypoint_format: str = "coco17",
        # END NEW
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            cam_int: Optional camera intrinsics [3, 3]
            keypoints_2d: Optional 2D keypoints [N, K, 2] or [N, K, 3] where K is number of keypoints
                          Coordinates should be in pixel space (not normalized)
                          If shape is [N, K, 3], third dimension is confidence score
            keypoints_3d: Optional 3D keypoints [N, K, 3] in camera coordinate system
            keypoint_scores: Optional keypoint confidence scores [N, K].
                            If keypoints_2d has shape [N, K, 3], this is ignored.
            keypoint_format: Format of keypoints (e.g., 'coco17', 'coco133', 'halpe26')
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]

        # NEW: Process keypoint inputs
        use_keypoints = keypoints_2d is not None
        if use_keypoints:
            keypoints_2d, keypoint_scores = self._process_keypoint_inputs(
                keypoints_2d, keypoint_scores, height, width
            )
            print(f"Using keypoint conditioning with {keypoint_scores.shape[1]} keypoints per person (format: {keypoint_format})")
        # END NEW

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            print("Found boxes:", boxes)
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # NEW: If keypoints are provided but no bboxes, create bboxes from keypoints
        if use_keypoints and bboxes is None and self.detector is None:
            print("Creating bounding boxes from keypoints...")
            boxes = self._create_bboxes_from_keypoints(keypoints_2d, keypoint_scores)
        # END NEW

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            assert (
                bboxes is not None
            ), "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img, boxes)
        else:
            masks, masks_score = None, None

        #################### Construct batch data samples ####################
        # NEW: Pass keypoints to prepare_batch
        batch = prepare_batch(
            img,
            self.transform,
            boxes,
            masks,
            masks_score,
            keypoints_2d=keypoints_2d if use_keypoints else None,
            keypoints_3d=keypoints_3d if use_keypoints else None,
            keypoint_scores=keypoint_scores if use_keypoints else None,
            keypoint_format=keypoint_format if use_keypoints else None,
        )
        # END NEW

        #################### Run model inference on an image ####################
        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        if cam_int is not None:
            print("Using provided camera intrinsics...")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image = batch["img_ori"][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()

        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                    "mhr_model_params": out["mhr_model_params"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out

    def _process_keypoint_inputs(
        self,
        keypoints_2d: np.ndarray,
        keypoint_scores: Optional[np.ndarray],
        height: int,
        width: int,
    ):
        """
        Process and validate keypoint inputs.

        Args:
            keypoints_2d: Input keypoints [N, K, 2] or [N, K, 3]
            keypoint_scores: Optional scores [N, K]
            height: Image height
            width: Image width

        Returns:
            Tuple of (processed_keypoints, scores)
        """
        # Ensure keypoints are in the right shape
        if len(keypoints_2d.shape) == 2:
            # Single person: [K, 2] or [K, 3]
            keypoints_2d = keypoints_2d[np.newaxis, ...]

        assert len(keypoints_2d.shape) == 3, \
            f"Keypoints should be [N, K, 2] or [N, K, 3], got {keypoints_2d.shape}"

        num_people, num_keypoints, coord_dim = keypoints_2d.shape

        # Extract confidence scores if they're in the keypoint array
        if coord_dim == 3:
            if keypoint_scores is None:
                keypoint_scores = keypoints_2d[:, :, 2]
            keypoints_2d = keypoints_2d[:, :, :2]

        # If no scores provided, use uniform confidence
        if keypoint_scores is None:
            keypoint_scores = np.ones((num_people, num_keypoints), dtype=np.float32)

        # Ensure scores are 2D
        if len(keypoint_scores.shape) == 1:
            keypoint_scores = keypoint_scores[np.newaxis, ...]

        # Validate keypoint coordinates are within image bounds
        keypoints_2d = np.clip(keypoints_2d, 0, [width, height])

        return keypoints_2d, keypoint_scores

    def _create_bboxes_from_keypoints(
        self,
        keypoints_2d: np.ndarray,
        keypoint_scores: np.ndarray,
        score_threshold: float = 0.3,
        padding_factor: float = 0.15,
    ):
        """
        Create bounding boxes from 2D keypoints.

        Args:
            keypoints_2d: Keypoints [N, K, 2]
            keypoint_scores: Scores [N, K]
            score_threshold: Minimum score for valid keypoint
            padding_factor: Padding to add around keypoints (as fraction of bbox size)

        Returns:
            Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        """
        num_people = keypoints_2d.shape[0]
        bboxes = []

        for i in range(num_people):
            kpts = keypoints_2d[i]
            scores = keypoint_scores[i]

            # Filter valid keypoints
            valid_mask = scores > score_threshold
            if not np.any(valid_mask):
                # If no valid keypoints, use all keypoints with lower confidence
                valid_mask = np.ones(len(scores), dtype=bool)

            valid_kpts = kpts[valid_mask]

            # Get bounding box
            x_min = np.min(valid_kpts[:, 0])
            y_min = np.min(valid_kpts[:, 1])
            x_max = np.max(valid_kpts[:, 0])
            y_max = np.max(valid_kpts[:, 1])

            # Add padding
            width = x_max - x_min
            height = y_max - y_min
            padding = padding_factor * max(width, height)

            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = x_max + padding
            y_max = y_max + padding

            bboxes.append([x_min, y_min, x_max, y_max])

        return np.array(bboxes)
