import os
from sys import argv
import nibabel as nib

from image_utils import save_array_to_nii

if __name__ == '__main__':
    data_path = argv[1]
    print(f'data_path:{data_path}')
    
    result_dir = argv[2]
    print(f'result_dir:{result_dir}')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    filename = os.path.basename(data_path)
    image_obj = nib.load(data_path)
    image = image_obj.get_fdata()

    save_array_to_nii(
        image,
        os.path.join(result_dir, filename),
        image_obj.affine,
    )
