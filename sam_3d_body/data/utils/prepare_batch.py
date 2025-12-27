# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.utils.data import default_collate
from typing import Optional


class NoCollate:
    def __init__(self, data):
        self.data = data


def prepare_batch(
    img,
    transform,
    boxes,
    masks=None,
    masks_score=None,
    cam_int=None,
    # NEW: Keypoint conditioning parameters
    keypoints_2d: Optional[np.ndarray] = None,
    keypoints_3d: Optional[np.ndarray] = None,
    keypoint_scores: Optional[np.ndarray] = None,
    keypoint_format: Optional[str] = None,
):
    """
    A helper function to prepare data batch for SAM 3D Body model inference.

    Args:
        img: Input image [H, W, 3]
        transform: Transform pipeline to apply
        boxes: Bounding boxes [N, 4] in xyxy format
        masks: Optional masks [N, H, W, 1]
        masks_score: Optional mask scores [N]
        cam_int: Optional camera intrinsics [3, 3]
        keypoints_2d: Optional 2D keypoints [N, K, 2] in pixel coordinates
        keypoints_3d: Optional 3D keypoints [N, K, 3]
        keypoint_scores: Optional keypoint confidence scores [N, K]
        keypoint_format: Format of keypoints (e.g., 'coco17', 'coco133')
    """
    height, width = img.shape[:2]

    # NEW: Validate keypoint inputs if provided
    has_keypoints = keypoints_2d is not None
    if has_keypoints:
        assert keypoints_2d.shape[0] == boxes.shape[0], \
            f"Number of keypoint sets ({keypoints_2d.shape[0]}) must match number of boxes ({boxes.shape[0]})"

        if keypoint_scores is None:
            # Default to uniform confidence if not provided
            num_keypoints = keypoints_2d.shape[1]
            keypoint_scores = np.ones((boxes.shape[0], num_keypoints), dtype=np.float32)

        print(f"Preparing batch with keypoint conditioning: {keypoints_2d.shape[1]} keypoints per person (format: {keypoint_format})")
    # END NEW

    # construct batch data samples
    data_list = []
    for idx in range(boxes.shape[0]):
        data_info = dict(img=img)
        data_info["bbox"] = boxes[idx]  # shape (4,)
        data_info["bbox_format"] = "xyxy"

        if masks is not None:
            data_info["mask"] = masks[idx].copy()
            if masks_score is not None:
                data_info["mask_score"] = masks_score[idx]
            else:
                data_info["mask_score"] = np.array(1.0, dtype=np.float32)
        else:
            data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
            data_info["mask_score"] = np.array(0.0, dtype=np.float32)

        # NEW: Add keypoint information to data_info
        if has_keypoints:
            data_info["keypoints_2d"] = keypoints_2d[idx].copy()  # [K, 2]
            data_info["keypoint_scores"] = keypoint_scores[idx].copy()  # [K]
            data_info["has_keypoints"] = True

            if keypoints_3d is not None:
                data_info["keypoints_3d"] = keypoints_3d[idx].copy()  # [K, 3]

            if keypoint_format is not None:
                data_info["keypoint_format"] = keypoint_format
        else:
            data_info["has_keypoints"] = False
        # END NEW

        data_list.append(transform(data_info))

    batch = default_collate(data_list)

    max_num_person = batch["img"].shape[0]

    # Unsqueeze and convert to float for standard keys
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()

    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)

    batch["person_valid"] = torch.ones((1, max_num_person))

    # NEW: Handle keypoint data in batch
    if has_keypoints:
        # Keypoints should be [N, K, 2] after collation, unsqueeze to [1, N, K, 2]
        if "keypoints_2d" in batch:
            batch["keypoints_2d"] = batch["keypoints_2d"].unsqueeze(0).float()
            print(f"Batch keypoints_2d shape: {batch['keypoints_2d'].shape}")

        if "keypoint_scores" in batch:
            batch["keypoint_scores"] = batch["keypoint_scores"].unsqueeze(0).float()
            print(f"Batch keypoint_scores shape: {batch['keypoint_scores'].shape}")

        if "keypoints_3d" in batch and keypoints_3d is not None:
            batch["keypoints_3d"] = batch["keypoints_3d"].unsqueeze(0).float()
            print(f"Batch keypoints_3d shape: {batch['keypoints_3d'].shape}")

        # Store metadata
        batch["has_keypoints"] = True
        if keypoint_format is not None:
            batch["keypoint_format"] = keypoint_format
    else:
        batch["has_keypoints"] = False
    # END NEW

    if cam_int is not None:
        batch["cam_int"] = cam_int.to(batch["img"])
    else:
        # Default camera intrinsics according image size
        batch["cam_int"] = torch.tensor(
            [
                [
                    [(height**2 + width**2) ** 0.5, 0, width / 2.0],
                    [0, (height**2 + width**2) ** 0.5, height / 2.0],
                    [0, 0, 1],
                ]
            ],
        ).to(batch["img"])

    batch["img_ori"] = [NoCollate(img)]
    return batch
