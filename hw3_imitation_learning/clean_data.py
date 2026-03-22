import zarr
import numpy as np
from pathlib import Path

# 1. Configuration
in_path = Path('datasets/processed/single_cube/processed_ee_xyz.zarr')
out_path = Path('datasets/processed/single_cube/processed_ee_xyz_cleaned.zarr')

print(f"Loading dataset from {in_path}...")
root = zarr.open_group(str(in_path), mode='r')
states = root['data/state_ee_xyz'][:]
ep_ends = root['meta/episode_ends'][:]
starts = np.insert(ep_ends[:-1], 0, 0)

# 2. Interpolate trajectories to 100 points so we can compare them
n_points = 100
trajs = []
for s, e in zip(starts, ep_ends):
    ep_data = states[s:e]
    t_orig = np.linspace(0, 1, len(ep_data))
    t_new = np.linspace(0, 1, n_points)

    # Interpolate X, Y, Z
    x = np.interp(t_new, t_orig, ep_data[:, 0])
    y = np.interp(t_new, t_orig, ep_data[:, 1])
    z = np.interp(t_new, t_orig, ep_data[:, 2])
    trajs.append(np.stack([x, y, z], axis=1))

trajs = np.array(trajs) # Shape: (Num Episodes, 100 points, 3 dims)

# 3. Compute the "Golden Path" (Median is better than Mean for ignoring crazy outliers)
median_traj = np.median(trajs, axis=0)

# Calculate average L2 distance of each episode from the median path
distances = np.linalg.norm(trajs - median_traj, axis=2).mean(axis=1)

print("\n--- Episode Distance Scores (Lower is closer to average) ---")
for i, d in enumerate(distances):
    print(f"Episode {i+1:2d}: {d:.5f}")

# 4. Set Threshold (Mean distance + 1 Standard Deviation)
threshold = np.mean(distances) + np.std(distances)
good_episodes = np.where(distances <= threshold)[0]
bad_episodes = np.where(distances > threshold)[0]

print(f"\nThreshold set to: {threshold:.5f}")
print(f"Keeping {len(good_episodes)} episodes. Dropping Episodes: {bad_episodes + 1}")

# 5. Build the mask of indices to KEEP
keep_mask = np.zeros(len(states), dtype=bool)
new_ep_ends = []
running_length = 0

for i in good_episodes:
    # Mark these rows as True to keep them
    keep_mask[starts[i]:ep_ends[i]] = True
    # Calculate new episode ends mathematically
    length = ep_ends[i] - starts[i]
    running_length += length
    new_ep_ends.append(running_length)

# 6. Write the new Cleaned Zarr Store
print(f"\nWriting cleaned dataset to {out_path}...")
out_root = zarr.open_group(str(out_path), mode='w', zarr_format=3)
out_root.attrs.update(root.attrs)
out_root.attrs['num_episodes'] = len(good_episodes)
out_root.attrs['num_transitions'] = int(np.sum(keep_mask))

out_data = out_root.require_group('data')
out_meta = out_root.require_group('meta')
compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)

# Copy and slice all data arrays
for key in root['data']:
    arr = root['data'][key][:]
    arr_filtered = arr[keep_mask]
    out_data.create_array(key, data=arr_filtered, compressors=(compressor,))

# Write the new episode ends
out_meta.create_array('episode_ends', data=np.array(new_ep_ends, dtype=np.int64), compressors=(compressor,))

print("Done! Cleaned dataset is ready for training.")