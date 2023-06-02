import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

def generate_orthogonal(vec):
    x = np.abs(vec[0])
    y = np.abs(vec[1])
    z = np.abs(vec[2])
    other = (np.array([1, 0, 0], dtype=vec.dtype) if x < z else np.array([0, 0, 1], dtype=vec.dtype)) if x < y else (np.array([0, 1, 0], dtype=vec.dtype) if y < z else np.array([0, 0, 1], dtype=vec.dtype))
    return np.cross(vec, other)

def get_quat_from_vec_to_vec(from_vec, to_vec):
    # x,y,z,w
    quat = np.zeros((4,), dtype=from_vec.dtype)
    if np.dot(from_vec, from_vec) == 0 or np.dot(to_vec, to_vec) == 0:
        quat[3] = 1
        return quat
    
    k_cos_theta = np.dot(from_vec, to_vec)
    k = np.sqrt(np.dot(from_vec, from_vec) * np.dot(to_vec, to_vec))

    if k_cos_theta / k == -1:
        # 180 degree rotation around any orthogonal vector.
        quat[3] = 0
        orthogonal = generate_orthogonal(from_vec)
        quat[0:3] = orthogonal / np.sqrt(np.dot(orthogonal, orthogonal))
    else:
        quat[3] = k_cos_theta + k
        quat[0:3] = np.cross(from_vec, to_vec)
        quat = quat / np.sqrt(np.dot(quat, quat))
    
    return quat

def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = None
    joint_parent = None
    joint_offset = None

    with open(bvh_file_path, "r") as bvh_file:
        lines = bvh_file.readlines()
        lines_count = len(lines)
        #print(bvh_file_path + " has %d lines."%lines_count)
        if lines_count > 1 and lines[0].strip().upper() == "HIERARCHY":
            joint_stack = []
            joint_name = []
            joint_parent = []
            joint_offset = []
            for lines_idx in range(1, lines_count):
                words = lines[lines_idx].strip().split()
                #print(lines_idx, words)
                words_count = len(words)
                if words_count > 0:
                    # Case root:
                    if words[0].upper() == "ROOT":
                        joint_parent.append(-1)
                        joint_stack.append(0)
                        joint_name.append("RootJoint")
                    # Case joint:
                    elif words[0].upper() == "JOINT":
                        joint_parent.append(joint_stack[-1])
                        joint_stack.append(len(joint_name))
                        joint_name.append(words[1])
                    # Case joint:
                    elif words[0].upper() == "END":
                        joint_parent.append(joint_stack[-1])
                        joint_stack.append(len(joint_name))
                        joint_name.append(joint_name[joint_parent[-1]] + "_end")
                    
                    # Case offset
                    elif words[0].upper() == "OFFSET":
                        joint_offset.append(np.array(
                            [float(words[1]),
                             float(words[2]),
                             float(words[3])]))
                    
                    # Case }
                    elif words[0].upper() == "}":
                        joint_stack.pop()

                    #print(words[0].upper(), joint_stack)

            joint_offset = np.array(joint_offset)


    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = None
    joint_orientations = None

    M = joint_offset.shape[0]
    cur_motion = motion_data[frame_id]
    joint_positions = np.zeros((M, 3), dtype=joint_offset.dtype)
    joint_orientations = np.zeros((M, 4), dtype=joint_offset.dtype)

    cur_joint_offsets = np.zeros_like(joint_positions)
    cur_joint_rotations = np.zeros_like(joint_orientations)

    motion_data_cursor = 0
    for joint_idx in range(0, M):
        # Root channel: trans(xyz) + rot(xyz).
        if joint_parent[joint_idx] == -1:
            #print(motion_data_cursor, cur_motion[motion_data_cursor:motion_data_cursor + 6])
            cur_joint_offsets[joint_idx] = joint_offset[joint_idx] + cur_motion[motion_data_cursor:motion_data_cursor + 3]
            cur_joint_rotations[joint_idx] = R.from_euler("XYZ", 
                                                         cur_motion[motion_data_cursor + 3:motion_data_cursor + 6], 
                                                         degrees=True).as_quat()
            motion_data_cursor += 6
        # End channel: null.
        elif joint_name[joint_idx].endswith('_end'):
            cur_joint_offsets[joint_idx] = joint_offset[joint_idx]
            cur_joint_rotations[joint_idx] = R.identity().as_quat()
        # Joint channel: rot(xyz).
        else:
            #print(motion_data_cursor, cur_motion[motion_data_cursor:motion_data_cursor + 3])
            cur_joint_offsets[joint_idx] = joint_offset[joint_idx]
            cur_joint_rotations[joint_idx] = R.from_euler("XYZ", 
                                                         cur_motion[motion_data_cursor:motion_data_cursor + 3], 
                                                         degrees=True).as_quat()
            motion_data_cursor += 3

    joint_computed = np.zeros((M,), dtype=np.uint8)
    def compute_joint_from_local_to_global(joint_idx : int):
        
        if joint_computed[joint_idx] > 0:
            return joint_positions[joint_idx], joint_orientations[joint_idx]

        cur_joint_position = cur_joint_offsets[joint_idx]
        cur_joint_orientation = cur_joint_rotations[joint_idx]
        if joint_parent[joint_idx] != -1:
            parent_joint_position, parent_joint_orientation = compute_joint_from_local_to_global(joint_parent[joint_idx])
            cur_joint_position = parent_joint_position + R(parent_joint_orientation).apply(cur_joint_position)
            cur_joint_orientation = (R(parent_joint_orientation) * R(cur_joint_orientation)).as_quat()
            
        joint_computed[joint_idx] = 1
        joint_positions[joint_idx] = cur_joint_position
        joint_orientations[joint_idx] = cur_joint_orientation
        return cur_joint_position, cur_joint_orientation

    for joint_idx in range(0, M):
        compute_joint_from_local_to_global(joint_idx)

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None

    joint_name_T, joint_parent_T, joint_offset_T = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_A, joint_parent_A, joint_offset_A = part1_calculate_T_pose(A_pose_bvh_path)
    
    joint_num = len(joint_name_A)
    joint_remap_from_T_to_A = np.zeros((joint_num,), dtype=int)
    for i in range(0, joint_num):
        joint_remap_from_T_to_A[i] = joint_name_A.index(joint_name_T[i])
    
    M = joint_offset_A.shape[0]

    motion_data_A = load_motion_data(A_pose_bvh_path)
    motion_data = np.zeros_like(motion_data_A)
    N = motion_data_A.shape[0]

    """
        Compute Q_{joint}^{a->t}
    """
    Q_A_to_T = np.zeros((M, 4), dtype=joint_offset_A.dtype)

    def compute_rest_positions(joint_name, joint_parent, joint_offset, joint_idx, 
                               out_joint_positions, inout_joint_computed):
        if inout_joint_computed[joint_idx] == 1:
            return out_joint_positions[joint_idx]
        
        position = joint_offset[joint_idx]
        if joint_parent[joint_idx] != -1:
            position += compute_rest_positions(joint_name, joint_parent, joint_offset, joint_parent[joint_idx],
                                              out_joint_positions, inout_joint_computed)

        out_joint_positions[joint_idx] = position
        inout_joint_computed[joint_idx] = 1
        return position

    joint_positions_T = np.zeros_like(joint_offset_T)
    joint_positions_A = np.zeros_like(joint_offset_A)
    joint_computed_T = np.zeros((M,), dtype=np.uint8)
    joint_computed_A = np.zeros((M,), dtype=np.uint8)
    for joint_idx_T in range(0, joint_num):
        joint_idx_A = joint_remap_from_T_to_A[joint_idx_T]
        compute_rest_positions(joint_name_T, joint_parent_T, joint_offset_T, joint_idx_T,
                               joint_positions_T, joint_computed_T)
        compute_rest_positions(joint_name_A, joint_parent_A, joint_offset_A, joint_idx_A,
                               joint_positions_A, joint_computed_A)
        
        if joint_parent_T[joint_idx_T] == -1:
            Q_A_to_T[joint_idx_T] = R.identity().as_quat()
        else:
            from_vec = joint_positions_A[joint_idx_A] - joint_positions_A[joint_parent_A[joint_idx_A]]
            to_vec = joint_positions_T[joint_idx_T] - joint_positions_T[joint_parent_T[joint_idx_T]]
            # Need to update parent Q...
            Q_A_to_T[joint_parent_T[joint_idx_T]] = get_quat_from_vec_to_vec(from_vec / np.sqrt(np.dot(from_vec, from_vec)),
                                                             to_vec / np.sqrt(np.dot(to_vec, to_vec)))
            if joint_name_T[joint_idx_T].endswith("_end"):
                Q_A_to_T[joint_idx_T] = R.identity().as_quat()

    # """TEST BEGIN"""
    # for joint_idx_T in range(0, joint_num):
    #     joint_idx_A = joint_remap_from_T_to_A[joint_idx_T]
    #     Q_parent_retarget = R.identity() if joint_parent_T[joint_idx_T] == -1 \
    #                 else R(Q_A_to_T[joint_parent_T[joint_idx_T]])
    #     Q_retarget = R(Q_A_to_T[joint_idx_T])
    #     rot_T = Q_parent_retarget * Q_retarget.inv()
    #     print(f"[GLOBAL] From A[{joint_name_A[joint_idx_A]}] to T[{joint_name_T[joint_idx_T]}]: Q = {Q_A_to_T[joint_idx_T]}, Q_euler = {R(Q_A_to_T[joint_idx_T]).as_euler('XYZ', degrees=True)}")
    #     #print(f"[LOCAL] From A[{joint_name_A[joint_idx_A]}] to T[{joint_name_T[joint_idx_T]}]: R = {rot_T.as_quat()}, R_euler = {rot_T.as_euler('XYZ', degrees=True)}")
    # """TEST END"""

    # R_i^{B} = Q_{p_i}^{A->B} @ R_i^{A} @ {Q_{i}^{A->B}}^T
    # R_i^{t} = Q_{p_i}^{a->t} @ R_i^{a} @ {Q_{i}^{a->t}}^T
    try:
        from tqdm import tqdm
    except:
        tqdm = lambda x: x
    for cur_frame in tqdm(range(0, N)):
        motion_data_cursor_T = 0
        positions_A, orientations_A = part2_forward_kinematics(joint_name_A, joint_parent_A, joint_offset_A, motion_data_A, cur_frame)
        for joint_idx_T in range(0, joint_num):
            joint_idx_A = joint_remap_from_T_to_A[joint_idx_T]
            
            rot_A = R(orientations_A[joint_idx_A]) if joint_parent_A[joint_idx_A] == -1 \
                    else R(orientations_A[joint_parent_A[joint_idx_A]]).inv() * R(orientations_A[joint_idx_A])
            Q_parent_retarget = R.identity() if joint_parent_T[joint_idx_T] == -1 \
                    else R(Q_A_to_T[joint_parent_T[joint_idx_T]])
            Q_retarget = R(Q_A_to_T[joint_idx_T])

            rot_T = Q_parent_retarget * rot_A * Q_retarget.inv()
            rot_T_euler = rot_T.as_euler("XYZ", degrees=True)
            
            # Root channel: trans(xyz) + rot(xyz).
            if joint_parent_T[joint_idx_T] == -1:
                #print(motion_data.shape, motion_data_cursor_T)
                motion_data[cur_frame, motion_data_cursor_T:motion_data_cursor_T + 3] = joint_positions_A[joint_idx_A] - joint_positions_T[joint_idx_T] + positions_A[joint_idx_A]
                motion_data[cur_frame, motion_data_cursor_T + 3:motion_data_cursor_T + 6] = rot_T_euler
                motion_data_cursor_T += 6
            # End channel: null.
            elif joint_name_T[joint_idx_T].endswith('_end'):
                pass
            # Joint channel: rot(xyz).
            else:
                motion_data[cur_frame, motion_data_cursor_T:motion_data_cursor_T + 3] = rot_T_euler
                motion_data_cursor_T += 3

    return motion_data
