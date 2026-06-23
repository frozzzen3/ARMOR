import os
import re

from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
import shutil

def get_dual_quaternions(original_centers, transformed_centers):
    moved_indices = []
    for i in range(len(original_centers)):
        if not (np.array_equal(original_centers[i], transformed_centers[i])):
            moved_indices.append(i)
    dual_quaternions = np.zeros((len(original_centers), 8), dtype=np.float32)
    inverse_dual_quaternions = np.zeros((len(original_centers), 8), dtype=np.float32)
    for i in moved_indices:
        original = original_centers[i]
        transformed = transformed_centers[i]

        rotation_quaternion = R.from_quat([0, 0, 0, 1])

        translation = transformed - original

        rotation_quat = rotation_quaternion.as_quat()

        translation_quat = np.hstack((translation, [0]))
        dual_quat = np.hstack((rotation_quat, 0.5 * translation_quat))
        dual_quaternions[i] = dual_quat

        rotation_conjugate = np.hstack((-rotation_quat[:3], rotation_quat[3]))
        inv_translation_quat = np.hstack((translation, [0]))
        inverse_dual_quat = np.hstack((rotation_conjugate, -0.5 * inv_translation_quat))
        inverse_dual_quaternions[i] = inverse_dual_quat

    return moved_indices, dual_quaternions, inverse_dual_quaternions


parser = argparse.ArgumentParser(description="Get transformation matrix.")

parser.add_argument('--centers_dir', type=str, required=True, help="Path for the volume centers")
parser.add_argument('--sourceIndex', type=int, required=True, help="source index")
parser.add_argument('--targetIndex', type=int, required=True, help="target index")


args = parser.parse_args()

centers_dir = args.centers_dir
sourceIndex = args.sourceIndex
targetIndex = args.targetIndex

output_dir = os.path.join(centers_dir, "transformation")


re_pattern = re.compile(r'(\d+)\.xyz$')

xyz_files_by_index = {}
for file_name in os.listdir(centers_dir):
    if not file_name.endswith('.xyz'):
        continue

    match = re_pattern.search(file_name)
    if not match:
        continue

    file_index = int(match.group(1))
    if file_index in xyz_files_by_index:
        raise ValueError(
            f"Multiple .xyz files match index {file_index}: "
            f"{xyz_files_by_index[file_index]}, {file_name}"
        )
    xyz_files_by_index[file_index] = file_name

available_indices = sorted(xyz_files_by_index)
missing_indices = [
    index for index in (sourceIndex, targetIndex)
    if index not in xyz_files_by_index
]
if missing_indices:
    raise ValueError(
        f"Missing .xyz file(s) for index {missing_indices}. "
        f"Available indices: {available_indices}"
    )

os.makedirs(output_dir, exist_ok=True)

loaded_centers_source = np.loadtxt(os.path.join(centers_dir, xyz_files_by_index[sourceIndex]))
loaded_centers_target = np.loadtxt(os.path.join(centers_dir, xyz_files_by_index[targetIndex]))
#print(centers_path)

indices, dual_quaternions, inverse_dual_quaternions = get_dual_quaternions(loaded_centers_source, loaded_centers_target)

indices_path = os.path.join(output_dir, f"indices_{sourceIndex:03}_{targetIndex:03}.txt")
np.savetxt(indices_path, indices, fmt='%d')

dual_quaternions_path = os.path.join(output_dir, f"transformations_{sourceIndex:03}_{targetIndex:03}.txt")
with open(dual_quaternions_path, 'w') as file:
    for dq in dual_quaternions:
        dq_str = f"{dq[0]};{dq[1]};{dq[2]};{dq[3]};{dq[4]};{dq[5]};{dq[6]};{dq[7]}"
        file.write(dq_str + '\n')


print("Centers transformations saved!")
print("Find centers transformations here: ", output_dir)
