from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import random

class PatchPool:

    def __init__(self, patches_folder: Path):
        self.patches_folder =  patches_folder
        self.failure_types = self.__getAvailableFailureTypes()
        # Load patches
        self.patches = self.__getPatches()

    def getRandomPatch(self, failure_type:str = None) -> Tuple[np.ndarray, str]:
        """
        Get a random patch, optionally from the specified failure type
        :param failure_type: (optional) failure type from the failure_types list. If None the failure type is chosen randomly
        :return: a tuple with a numpy ndarray representing an image in RGBA format and the corresponding failure type
        """
        failure = failure_type if failure_type else random.sample(self.failure_types, 1)[0]
        return random.sample(self.patches[failure], 1)[0].copy(), failure

    ## UTILS ##

    def __getAvailableFailureTypes(self) -> List[str]:
        failure_types = []
        for pp in self.patches_folder.iterdir():
            if pp.is_dir() and not pp.parts[-1].endswith("__"):
                failure_types.append(pp.parts[-1])
        return failure_types

    def __loadFailurePatches(self, failure_type: str) -> List[np.ndarray]:
        f_patches = []
        folder_p = self.patches_folder/failure_type
        for patch_p in folder_p.glob("*.png"):
            img = cv2.imread(str(patch_p), cv2.IMREAD_UNCHANGED)        # BGRA
            f_patches.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))    # RGBA
        return f_patches

    def __getPatches(self) -> Dict:
        patches_dict = {}
        for fail_name in self.failure_types:
            patches_dict[fail_name] = self.__loadFailurePatches(fail_name)
        return patches_dict

