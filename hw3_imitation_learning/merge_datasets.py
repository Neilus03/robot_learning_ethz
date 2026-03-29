import zarr
import numpy as np

# 1. Define our paths
clean_path = 'datasets/processed/single_cube/processed_ee_xyz_cleaned.zarr'
dagger_path = 'datasets/processed/single_cube/processed_ee_xyz.zarr'
out_path = 'datasets/processed/single_cube/processed_ee_xyz_final_cleaned.zarr'

print(f"Loading Base Data: {clean_path}")
clean_root = zarr.open_group(clean_path, mode='r')

print(f"Loading DAgger Data: {dagger_path}")
dagger_root = zarr.open_group(dagger_path, mode='r')

# 2. Setup the new combined dataset
print(f"Creating Merged Dataset: {out_path}")
out_root = zarr.open_group(out_path, mode='w', zarr_format=3)
out_root.attrs.update(clean_root.attrs)
out_root.attrs['num_episodes'] = clean_root.attrs['num_episodes'] + dagger_root.attrs['num_episodes']
out_root.attrs['num_transitions'] = clean_root.attrs['num_transitions'] + dagger_root.attrs['num_transitions']

out_data = out_root.require_group('data')
out_meta = out_root.require_group('meta')
compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)

# 3. Concatenate all state and action arrays
for key in clean_root['data']:
    arr1 = clean_root['data'][key][:]
    arr2 = dagger_root['data'][key][:]
    merged_arr = np.concatenate([arr1, arr2], axis=0)
    out_data.create_array(key, data=merged_arr, compressors=(compressor,))

# 4. Mathematically shift the episode ends for the DAgger data so they align
ep_ends1 = clean_root['meta/episode_ends'][:]
ep_ends2 = dagger_root['meta/episode_ends'][:]

offset = ep_ends1[-1] # The length of the first dataset
ep_ends2_shifted = ep_ends2 + offset
merged_ep_ends = np.concatenate([ep_ends1, ep_ends2_shifted], axis=0)

out_meta.create_array('episode_ends', data=merged_ep_ends, compressors=(compressor,))

print("\n=== MERGE SUCCESSFUL ===")
print(f"Base Episodes:   {clean_root.attrs['num_episodes']}")
print(f"DAgger Episodes: {dagger_root.attrs['num_episodes']}")
print(f"Total Episodes in Final Dataset: {out_root.attrs['num_episodes']}")