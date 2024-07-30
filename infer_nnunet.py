import os
import glob
import torch

from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import SimpleITK as sitk
from tqdm import tqdm


def get_ml_sitk(image):
    filter = sitk.StatisticsImageFilter()
    filter.Execute(image)
    spacing = image.GetSpacing()
    voxel = spacing[0] * spacing[1] * spacing[2]
    cubic_mm = filter.GetSum() * voxel
    ml = cubic_mm / 1000
    return ml


if __name__ == '__main__':
    nnUNet_results = r'nnUNet_results_100'
    nnUNet_raw = r'nnUNet_raw'
    nnUNet_preprocessed = r'nnUNet_preprocessed'

    indir = join(nnUNet_raw, r'Dataset022_niddkitaly/imagesTs')
    outdir = join(nnUNet_raw, r'Dataset022_niddkitaly/imagesTs/predictions')

    # Check if CUDA is available, and use CPU if it is not
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
        perform_everything_on_gpu = True
        print("CUDA is available. Using GPU for inference.")
    else:
        device = torch.device('cpu')
        perform_everything_on_gpu = False
        print("CUDA is not available. Using CPU for inference.")

    # Instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=perform_everything_on_gpu,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, r'Dataset022_niddkitaly/nnUNetTrainer_100epochs__nnUNetPlans__2d'),
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name='checkpoint_final.pth',
    )

    nii_files = glob.glob(os.path.join(indir, '*.nii.gz'))
    for infile in tqdm(nii_files):
        filename = os.path.basename(infile)
        outfile = join(outdir, filename)
        predictor.predict_from_files([[infile]],
                                     [outfile],
                                     save_probabilities=False, overwrite=False,
                                     num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        seg_sitk = sitk.ReadImage(outfile)
        label_1 = seg_sitk == 1
        label_2 = seg_sitk == 2
        vol_1 = get_ml_sitk(label_1)
        vol_2 = get_ml_sitk(label_2)
        print(f'****************** prediction finished for {infile} ******************')
        print(f'!!!!!!!!!!!!!!!!!! volume of label 1: {vol_1} ml !!!!!!!!!!!!!!!!!!'
              f'\n!!!!!!!!!!!!!!!!!! volume of label 2: {vol_2} ml !!!!!!!!!!!!!!!!!!')
    print('****************** Inference complete ******************')