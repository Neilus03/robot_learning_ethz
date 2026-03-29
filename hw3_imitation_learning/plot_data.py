import zarr
import plotly.graph_objects as go
import numpy as np

# Load your processed data
root = zarr.open('datasets/processed/single_cube/processed_ee_xyz_cleaned.zarr', mode='r')
states = root['data/state_ee_xyz'][:]
episode_ends = root['meta/episode_ends'][:]

fig = go.Figure()
start_idx = 0

for i, end_idx in enumerate(episode_ends):
    ep_states = states[start_idx:end_idx]
    x, y, z = ep_states[:, 0], ep_states[:, 1], ep_states[:, 2]

    # Add each episode as a separate interactive trace
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        name=f'Episode {i+1}',
        opacity=0.7
    ))
    start_idx = end_idx

fig.update_layout(title='End-Effector Trajectories', margin=dict(l=0, r=0, b=0, t=40))
fig.show()