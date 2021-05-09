
import numpy as np


def dictionary(letter):
    """
    For each letter (D, H, T), creates a matrix (path) of via points (x,y,z) and a vector (extrusion) of extrusions
    between consecutive via points. Extrusion is a vector of boolean values, where 1 (True) means extrude,
    and 0 (False) not extrude.
    inputs
        letter: char
    returns
        path: numpy array, float, shape (N_via_points, 3)
        extrusion: numpy array, boolean, shape (N_via_points)
    """

    path = []
    extrusion = []
    # TODO
    # compute the paths and extrusions for the letters D, H, T
    init_head = [0, 0, 1]
    start = [0, 0, 0]
    end = start

    if letter == 'D':
        path = np.array([
            init_head, start, [0, 2, 0], [0.6, 2, 0], [1, 1.6, 0], [1, 0.4, 0],
            [0.6, 0, 0], [0.4, 0, 0], [0.4, 0.2, 0], [0.6, 0.2, 0], [0.8, 0.4, 0],
            [0.8, 1.6, 0], [0.6, 1.8, 0], [0.2, 1.8, 0], [0.2, 0, 0], end, init_head
        ])
        extrusion = np.array([False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False])
    elif letter == 'H':
        path = np.array([
            init_head, start, [0, 2, 0], [0.2, 2, 0], [0.2, 1.1, 0], [0.8, 1.1, 0],
            [0.8, 2, 0], [1, 2, 0], [1, 0, 0], [0.8, 0, 0], [0.8, 0.9, 0], [0.2, 0.9, 0],
            [0.2, 0, 0], end, init_head
        ])
        extrusion = np.array(
            [False, True, True, True, True, True, True, True, True, True, True, True, True, False, False])
    elif letter == 'T':
        path = np.array([
            init_head, start, [0, 1.8, 0], [0, 2, 0], [1, 2, 0], [1, 1.8, 0], [0.6, 1.8, 0],
            [0.6, 0, 0], [0.4, 0, 0], [0.4, 1.8, 0], [0, 1.8, 0], end, init_head
        ])
        extrusion = np.array(
            [False, False, True, True, True, True, True, True, True, True, False, False, False])

    return path, extrusion


def compute_transformation_dh(theta, d, a, alpha):
    """
    Computes the transformation matrix between joint i to joint i+1.
    given the DH-Parameters alpha, a, d, theta.
    inputs
        theta: DH-Parameter theta_i, float
        d: DH-Parameter d_i, float
        a: DH-Parameter a_i, float
        alpha: DH-Parameter alpha_i, float
    returns
        transformation_matrix: numpy array, float, shape (4, 4)
    """

    # TODO
    # compute the transformation matrix
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)

    trans_matrix = np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

    return trans_matrix


def compute_forward_kinematic(q, joint_limits):
    """
    Computes the end-effector coordinates (x,y,z) with respect to the frame Sp, for the given joint displacements q.
    q is a vector with the joint displacements.
    joint_limits is an array with the minimum and maximum joint values for each joint in q.
    inputs
        q: numpy array, float, shape (3,1)
        joint_limits: numpy array, float, shape (3, 2)
    returns
        position: numpy array, float, shape (3,1)
    """
    # Check if requested joint configuration does not violate joint limits
    q_min = joint_limits[:, 0]
    q_max = joint_limits[:, 1]

    e = 1e-6

    for i in range(len(q)):
        if q[i] < q_min[i] - e or q[i] > q_max[i] + e:
            return []

    # Compute forward kinematics using DH transformations
    # transformation p_T_pa (pa is a dummy point between p and 0)
    p_T_pa = compute_transformation_dh(np.pi / 2, 0, -0.02, 0)
    # transformation pa_T_0
    pa_T_0 = compute_transformation_dh(-np.pi / 2, 0.07, -0.02, 0)
    # transformation p_T_0
    p_T_0 = np.matmul(p_T_pa, pa_T_0)

    # TODO
    # Compute forward kinematics using DH transformations p_T_3
    # attention: p_T_0 is already computed
    zero_T_1 = compute_transformation_dh(np.pi/2,q[0], 0.02, np.pi/2)
    one_T_2 = compute_transformation_dh(-np.pi/2,q[1] + 0.02, 0.02, -np.pi/2)
    two_T_3 = compute_transformation_dh(0, q[2], 0.048, 0)

    p_T_1 = np.matmul(p_T_0, zero_T_1)
    p_T_2 = np.matmul(p_T_1, one_T_2)
    p_T_3 = np.matmul(p_T_2, two_T_3)

    p_r_3 = p_T_3[3, 0:3]
    return p_r_3.reshape((3,1))


def compute_inverse_kinematic(position, joint_limits):
    """
    Computes the joint displacements given the position of the end-effector.
    position is a vector with the (x,y,z) coordinates of the end-effector position in the frame Sp.
    joint_limits is an array with the maximum and minimum joint values for each joint in q.
    inputs
        position: numpy array, float, shape (3,1)
        joint_limits: numpy array, float, shape (3, 2)
    returns
        q: numpy array, float, shape (3,1)
    """
    # Check if requested position is in workspace
    q_min = joint_limits[:, 0]
    q_max = joint_limits[:, 1]

    position_min = compute_forward_kinematic(
        q_min,
        joint_limits
    )
    position_max = compute_forward_kinematic(
        q_max,
        joint_limits
    )

    e = 1e-6
    for i in range(len(position)):
        if position[i] < position_min[i] - e \
                or position[i] > position_max[i] + e:
            return []

    # TODO
    # Compute analytical inverse kinematics
    def f(q):
        return position - compute_forward_kinematic(q, joint_limits)

    def jacobian(q, fq):
        h = 0.0001
        q_next = q + h
        q_next_1 = np.array([q_next[0], q[1], q[2]])
        q_next_2 = np.array([q[0], q_next[1], q[2]])
        q_next_3 = np.array([q[0], q[1], q_next[2]])

        J = np.column_stack((f(q_next_1)-fq, f(q_next_2)-fq, f(q_next_3)-fq)) / h
        return J

    q_before = q_min
    fq = f(q_before)
    q_next = q_before - np.matmul(np.linalg.inv(jacobian(q_before, fq)), fq)
    is_converge = all(abs(q_next - q_before) < e)

    while not is_converge:
        q_before = q_next
        fq = f(q_before)
        q_next = q_before - np.linalg.inv(jacobian(q_before, fq)).dot(fq)
        is_converge = all(abs(q_next - q_before) < e)

    return q_next


def compute_letter_area(points):
    """
    Computes the total area of a letter using the given points of path.
    points is an array of points, every point has the shape (1,3).
    inputs
        points: numpy array, float, with shape (N_via_points, 3)
    returns
        area: float
    """
    # TODO
    # compute letter area

    return None