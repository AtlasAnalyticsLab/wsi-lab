# Third-party software

This repository has no owner-supplied license. Nothing in this file grants rights to the project itself.

AtlasPatch 1.1.0 is pinned because it matches the embedding provenance and supplies all tissue detection. Its upstream repository identifies the project under Creative Commons Attribution-NonCommercial-ShareAlike 4.0. Review its terms before commercial use or redistribution: <https://github.com/AtlasAnalyticsLab/AtlasPatch>.

AtlasPatch tissue detection obtains `model.pth` from the
`AtlasAnalyticsLab/AtlasPatch` Hugging Face repository. The exact resolved
snapshot and checkpoint SHA256 are runtime provenance recorded in each
generated tissue manifest; review the model repository's terms before use or
redistribution.

SAM2 is installed at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` from the official repository and is required by AtlasPatch WSI segmentation: <https://github.com/facebookresearch/sam2>.

OpenSlide performs WSI reads: <https://openslide.org/>. PyTorch, torchvision, Gymnasium, NumPy, HDF5/h5py, Pydantic, PyYAML, SciPy, scikit-learn, Pillow, Matplotlib, TensorBoard, psutil, and optional W&B remain subject to their respective upstream licenses.

CleanRL is a design reference for recurrent PPO semantics only. It is not a runtime dependency and no CleanRL source package is vendored: <https://github.com/vwxyzjn/cleanrl>.
