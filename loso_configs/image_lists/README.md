# image_lists/

This directory is intentionally empty. The per-source train/val/test image lists
referenced by `loso_configs/loso/*.yaml` and `loso_configs/single_source/*.yaml`
contain absolute paths to patient X-ray images and are **not** redistributed
in the public repository because of patient-privacy restrictions.

To regenerate them, place the de-identified dataset on disk and run:

    python step0_prepare_data.py --data_root <path-to-dataset>
    python step1_make_loso_splits.py

This will populate `image_lists/` with `loso_<source>_{train,val,test}.txt`
and `single_<source>_{train,val,test}.txt`.
