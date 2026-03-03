import numpy as np
import mujoco


def get_lemniscate_keypoint(t, a=0.2):
    """
    TODO:
    Generate a set of keypoints using Lemniscate of Bernoulli (infinity sign) in the Y-Z plane.
        The formula is: y = a * cos(t) / (1 + sin(t)^2)
                        z = a * cos(t) * sin(t) / (1 + sin(t)^2)
    For interest, you can learn about Lemniscate of Bernoulli on wikipedia: https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli

    Args:
        t (float or np.ndarray): Time scales from 0 to 2π to generate keypoints.
        a (float): Scaling factor for the size of the lemniscate.

    Returns:
        y (float or np.ndarray): y coordinates of the keypoint on the lemniscate.
        z (float or np.ndarray): z coordinates of the keypoint on the lemniscate.
    """
    y = a * np.cos(t) / (1 + np.sin(t)**2)
    z = a * np.cos(t) * np.sin(t) / (1 + np.sin(t)**2)
    return y, z

def build_keypoints(count=16, width=0.25, x_offset=0.3, z_offset=0.25):
    """TODO:
    Build a set of keypoints (x, y, z) along the lemniscate trajectory.
    Steps:
    1. Generate `count` linearly spaced time values `t` between 0 and 2π (exclusive).
    2. For each time value `t`, compute the corresponding (y, z) coordinates using `get_lemniscate_keypoint(t, a=width)`.
    3. Combine the (y, z) coordinates with a fixed x coordinate (x_offset) and additive z_offset to create 3D keypoints in the format [x_offset, y, z + z_offset].
    4. Return the keypoints as a NumPy array of shape (count, 3).

    Args:
        count (int): Number of keypoints to generate along the trajectory.
        width (float): Scaling factor for the size of the lemniscate.
        x_offset (float): Fixed x coordinate for all keypoints.
        z_offset (float): Offset to add to the z coordinate of all keypoints.

    Returns:
        np.ndarray: Array of shape (count, 3) containing the generated keypoints.
    """
    # Generate 'count' linearly spaced time values between 0 and 2π (exclusive)
    t = np.linspace(0, 2 * np.pi, count, endpoint=True)

    # Compute (y, z) coordinates using get_lemniscate_keypoint
    y, z = get_lemniscate_keypoint(t, a=width)

    # Create 3D keypoints with fixed x_offset and z_offset
    keypoints = np.zeros((count, 3))
    for i in range(count):
        keypoints[i, 0] = x_offset      # x coordinate
        keypoints[i, 1] = y[i]          # y coordinate
        keypoints[i, 2] = z[i] + z_offset  # z coordinate with offset

    return keypoints

def ik_track(model, data, site_name, target_pos,
             damping=1e-3, pos_gain=2.0, dt=0.1, max_iters=2000):
    """TODO:
    Implement an IK tracking function that computes the joint configuration to reach a target end-effector position. We ignore orientation tracking for simplicity.
    The function should iteratively update the joint configuration using the Jacobian of the end-effector until it reaches the target within a specified tolerance
    or exceeds the maximum number of iterations. We use the Damped Least Squares method to handle singularities in the Jacobian. For interest, you can learn about
    Damped Least Squares method on wikipedia: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

    Steps:
    1. Store the original joint configuration (qpos) to restore later.
    2. For a maximum number of iterations:
        a. Compute the current end-effector position and orientation using forward kinematics (mj_kinematics).
        b. Calculate the position error (target_pos - current_pos).
        c. If the position error is below a certain threshold (e.g., 1e-3), break the loop as we have reached the target.
        d. Compute the Jacobian of the end-effector using mj_jacSite.
        e. Use the Damped Least Squares to compute the change in joint configuration (qdot) that would reduce the error.
        f. Update the joint configuration (qpos) using the output from the Damped Least Squares method.
    3. Restore the original joint configuration and return the target joint configuration that was computed.

    Args:
        model: MuJoCo model object.
        data: MuJoCo data object.
        site_name: Name of the end-effector site to track.
        target_pos: Desired position of the end-effector (3D vector).
        damping: Damping factor for the Damped Least Squares method to handle singularities.
        pos_gain: Gain factor for the position error in the control signal.
        dt: Time step for updating the joint configuration.
        max_iters: Maximum number of iterations to attempt for reaching the target.

    Returns:
        np.ndarray: Target joint configuration (qpos) that achieves the desired end-effector position.
    """
    num_joints = model.nv
    # Store the original joint configuration to restore later
    original_qpos = data.qpos.copy()

    for i in range(max_iters):
        # use forward kinematics to update current end-effector position: data.site(site_name).xpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # TODO: compute end-effector position error
        err_pos = target_pos - data.site(site_name).xpos

        # TODO: check if the 2-norm of the position error is within a small threshold (1e-3), if yes, break the loop
        if np.linalg.norm(err_pos) < 1e-3:
            break

        # Get the Jacobian of the end-effector using mj_jacSite.
        jacp = np.zeros((3, num_joints)) # position Jacobian
        jacr = np.zeros((3, num_joints)) # orientation Jacobian
        mujoco.mj_jacSite(model, data, jacp, jacr, model.site(site_name).id)
        J = np.vstack([jacp, jacr])  # shape (6, nv)

        # TODO: compute the change in joint configuration (qdot) using Damped Least Squares method to reduce the position error
        # Damped least squares: qdot = J^T @ (J @ J^T + damping * I)^-1 @ weighted_err
        # Hint: damping * I is a 6x6 matrix with damping on the diagonal, and weighted error is a 6D vector (3 for pos, 3 for rot) of the form
        # [pos_gain * err_pos, rot_gain * err_rot]. Since we are ignoring orientation tracking, you can set the rotational part of the weighted error to zero.
        # Instead of directly computing the matrix inverse (which can be numerically unstable), you should use np.linalg.solve to solve the
        # linear system (J @ J^T + damping * I) x = weighted_err for x, and then compute qdot = J^T @ x. This is more stable and efficient than computing the inverse.
        weighted_err = np.concatenate([pos_gain * err_pos, np.zeros(3)])
        A = J @ J.T + damping * np.eye(6)
        x = np.linalg.solve(A, weighted_err)
        qdot = J.T @ x
        data.qpos[:] += qdot * dt
    # Restore the original joint configuration and return the target joint configuration
    target_qpos = data.qpos.copy()
    data.qpos[:] = original_qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_forward(model, data)
    return target_qpos

# Theoretical questions:
# - If you increase the width of the Lemniscate (increasing a), what issue can happen with the robot performing IK?
#     The target trajectory becomes wider and taller, therefore,the robot doesn't arrive at the target position (unreachability) because the end-effector is too far away from the target position. Also some Kinematic Singularities can happen because the robot is too far away from the target position.
#
# - What can happen if you change the dt parameter in IK?
#     dt acts as the step size or learning rate for the iterative solver, so if it is too large, it overshoots the target position and doesn't converge to the target position, when i set it a very small value, it converges to the target position but it takes too long to converge.
#
# - We implemented a simple numerical IK solver. What are the advantages and disadvantages compared to an analytical IK solver?
#     Advantage: Numerical IK solvers are highly versatile and easily handle complex, redundant robots without requiring custom algebraic derivations for every specific robot model.
#     Disadvantage: They are computationally slower, prone to getting stuck in local minima, and only find one approximate solution rather than all exact configurations provided by analytical solvers.
#
# - What are the limits of our IK solver compared to state-of-the-art IK solvers?
#     This implementation ignores:
#         - orientation tracking: It doesn't take into account the orientation of the end-effector, therefore, it might be at the right position but the wrong orientation.
#         - joint limits: might output a joint angle of $400^\circ$ to reach a target, physically breaking the real robot
#         - collision avoidance: might output a joint angle that is in collision with the environment
#         - null space exploitation: only optimizes for end effector position, not for joint angles or other constraints
#         - etc.
#     Also it is susceptible to local minima and numerical instability.