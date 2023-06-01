from pathlib import Path
import albumentations as A
import cv2
import numpy as np
from PIL import Image
from albumentations import PadIfNeeded

from PatchPool import PatchPool


class AugmentedPatchInjector:

    def __init__(self, patches_folder: Path, target_height: int, target_width: int):
        self.patch_pool = PatchPool(patches_folder)
        self.transformations = self.__getTransformations(target_height, target_width)

    def inject(self, target: np.ndarray, failure_type:str = None) -> np.ndarray:
        """
        Inject a randomly augmented patch to the target image
        :param target: numpy array representing an image in GRAYSCALE format
        :param failure_type: (optional) to specify the patch failure type
        :return: the injected image, in the same GRAYSCALE format
        """
        # Get patch
        patch_np, failure = self.patch_pool.getRandomPatch(failure_type)
        # Augment patch
        transformed = self.transformations[failure](image=patch_np)
        augmented_patch_np = transformed['image']
        # Apply patch
        injected = self.__applyPILPatch(target, augmented_patch_np)
        return injected

    ### UTILS ###

    def __getTransformations(self, t_height: int, t_width: int):
        return {
            "rain": A.Compose([
                A.ShiftScaleRotate(scale_limit=(-0.5, 0.5), rotate_limit=90, border_mode=cv2.BORDER_WRAP,
                                   always_apply=True),
                A.LongestMaxSize(max_size=min((t_height, t_width)), always_apply=True),
                A.Rotate(border_mode=cv2.BORDER_WRAP, always_apply=True),
                A.PadIfNeeded(min_height=t_height, min_width=t_width,
                              position=PadIfNeeded.PositionType.RANDOM, border_mode=cv2.BORDER_WRAP, always_apply=True),
            ]),

            "ice": A.Compose([
                A.ShiftScaleRotate(scale_limit=(-0.5, 0.5), rotate_limit=90, border_mode=cv2.BORDER_REFLECT_101,
                                   always_apply=True),
                A.LongestMaxSize(max_size=min((t_height, t_width)), always_apply=True),
                A.Rotate(border_mode=cv2.BORDER_REFLECT_101, always_apply=True),
                A.OneOf([
                    # Small patch
                    A.Compose([
                        A.PadIfNeeded(min_height=t_height, min_width=t_width,
                                      position=PadIfNeeded.PositionType.RANDOM, border_mode=cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255, 0), always_apply=True),
                        A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255, 0), always_apply=True),
                    ], p=0.8),
                    # Big patch
                    A.Compose([
                        A.PadIfNeeded(min_height=t_height, min_width=t_width,
                                      position=PadIfNeeded.PositionType.RANDOM, border_mode=cv2.BORDER_WRAP,
                                      always_apply=True),
                        A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255, 0), p=0.3),
                    ], p=0.2),
                ], p=1.0),
            ]),

            "condensation": A.Compose([
                A.ShiftScaleRotate(scale_limit=(-0.5, 0.5), rotate_limit=90, border_mode=cv2.BORDER_REFLECT_101,
                                   always_apply=True),
                A.LongestMaxSize(max_size=min((t_height, t_width)), always_apply=True),
                A.OneOf([
                    # Small patch
                    A.Compose([
                        A.PadIfNeeded(min_height=t_height, min_width=t_width,
                                      position=PadIfNeeded.PositionType.RANDOM, border_mode=cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255, 0), always_apply=True),
                    ], p=0.2),
                    # Big patch
                    A.Compose([
                        A.PadIfNeeded(min_height=t_height, min_width=t_width,
                                      position=PadIfNeeded.PositionType.RANDOM, border_mode=cv2.BORDER_REFLECT_101,
                                      always_apply=True),
                        A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255, 0), p=0.3),
                    ], p=0.8),
                ], p=1.0),
            ]),

            "breakage": A.Compose([
                A.ShiftScaleRotate(scale_limit=(-0.5, 0.5), rotate_limit=90, border_mode=cv2.BORDER_REFLECT_101,
                                   always_apply=True),
                A.Flip(p=0.3),
                A.Transpose(p=0.3),
                A.LongestMaxSize(max_size=min((t_height, t_width)), always_apply=True),
                A.Rotate(border_mode=cv2.BORDER_REFLECT_101, always_apply=True),
                A.OneOf([
                    # Small patch
                    A.Compose([
                        A.PadIfNeeded(min_height=t_height, min_width=t_width,
                                      position=PadIfNeeded.PositionType.RANDOM, border_mode=cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255, 0), always_apply=True),
                        A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255, 0), always_apply=True),
                    ], p=0.8),
                    # Big patch
                    A.Compose([
                        A.PadIfNeeded(min_height=t_height, min_width=t_width,
                                      position=PadIfNeeded.PositionType.RANDOM, border_mode=cv2.BORDER_REFLECT_101,
                                      always_apply=True),
                        A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255, 0), p=0.5),
                    ], p=0.2),
                ], p=1.0),
            ]),

            "dirt": A.Compose([
                A.ShiftScaleRotate(scale_limit=(-0.1, 0.1), rotate_limit=90, border_mode=cv2.BORDER_REFLECT_101,
                                   always_apply=True),
                A.Flip(p=0.3),
                A.Transpose(p=0.3),
                A.LongestMaxSize(max_size=min((t_height, t_width)), always_apply=True),
                A.Rotate(border_mode=cv2.BORDER_REFLECT_101, always_apply=True),
                A.OneOf([
                    # Small patch
                    A.Compose([
                        A.PadIfNeeded(min_height=t_height, min_width=t_width,
                                      position=PadIfNeeded.PositionType.RANDOM, border_mode=cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255, 0), always_apply=True),
                        A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255, 0), always_apply=True),
                    ], p=0.3),
                    # Big patch
                    A.Compose([
                        A.PadIfNeeded(min_height=t_height, min_width=t_width,
                                      position=PadIfNeeded.PositionType.RANDOM, border_mode=cv2.BORDER_REFLECT_101,
                                      always_apply=True),
                        A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255, 0), p=0.3),
                    ], p=0.7),
                ], p=1.0),
            ])
        }

    # def __put4ChannelImageOn4ChannelImage(self, back, fore, x, y):
    #     rows, cols, channels = fore.shape
    #     trans_indices = fore[..., 3] != 0  # Where not transparent
    #     overlay_copy = back[y:y + rows, x:x + cols]
    #     overlay_copy[trans_indices] = fore[trans_indices]
    #     back[y:y + rows, x:x + cols] = overlay_copy
    #
    # def __applyPatch(self, target: np.ndarray, augmented_patch_np: np.ndarray) -> np.ndarray:
    #     # Convert target to RGBA format
    #     target_rgba = cv2.cvtColor(target, cv2.COLOR_GRAY2RGBA)
    #     # Apply patch
    #     self.__put4ChannelImageOn4ChannelImage(target_rgba, augmented_patch_np, 0,0)
    #     # Convert back to grayscale
    #     injected = cv2.cvtColor(target_rgba, cv2.COLOR_RGBA2GRAY)
    #     return injected

    def __applyPILPatch(self, target: np.ndarray, augmented_patch_np: np.ndarray) -> np.ndarray:
        # Convert to PIL images
        target_img = Image.fromarray(target).convert("RGBA")
        patch_img = Image.fromarray(augmented_patch_np)
        # Apply Patch
        patched = target_img.copy()
        patched.paste(patch_img, (0, 0), patch_img)
        # Convert back to grayscale
        patched = patched.convert("L")
        # Convert to cv2 image
        # noinspection PyTypeChecker
        patched_np = np.array(patched)
        return patched_np

