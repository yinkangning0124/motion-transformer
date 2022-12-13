from typing import Tuple
import scann
import torch
import numpy as np
import pyrender
import smplx
import trimesh
import os
import cv2
from torch import Tensor

os.environ["PYOPENGL_PLATFORM"] = "egl"


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # compute exponential map from axis-angle
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # compute exponential map from quaternion
    # q must be normalized
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


def obs_visualize(obs):
    """
    obs numpy shape (1, 4+63) root_rot + dof_pos

    #save obs for experiment
    obs_save = torch.cat((self._humanoid_root_states[:1, 3:7], self._dof_pos[:1]), dim=-1)
    obs_save = obs_save.detach().cpu().numpy()
    import datetime
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    obs_folder = os.path.abspath("obs/new")
    # Create output folder if needed
    os.makedirs(obs_folder, exist_ok=True)
    savepath = os.path.join(obs_folder, time_str+".npy")
    np.save(savepath, obs_save)

    """
    DOF_BODY_IDS = [
        3,
        6,
        9,
        13,
        16,
        18,
        20,
        12,
        15,
        14,
        17,
        19,
        21,
        2,
        5,
        8,
        11,
        1,
        4,
        7,
        10,
    ]
    body_model = "smpl"
    body_model_path = "./assets/smpl_model/models"  # share drive

    obs = torch.from_numpy(obs).view(-1, 1, 67) # shape [batchsize, 1, 67]

    rot = torch.zeros(list(obs.shape[:-2]) + [24, 3]) # shape [batchsize, 24, 3]
    rot[..., DOF_BODY_IDS, :] = obs[..., 4:].view(list(obs.shape[:-2]) + [21, 3])
    rot[..., 0, :] = quat_to_exp_map(obs[:, 0, :4])

    body_model = smplx.create(model_path=body_model_path, model_type=body_model)
    faces = body_model.faces

    vertices = (
        body_model(global_orient=rot[..., :1, :], body_pose=rot[..., 1:, :])
        .vertices[0]
        .detach()
        .numpy()
    )
    # vertices = body_model().vertices[0].detach().numpy()

    original_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # original_mesh.export('obsvistest.ply')
    mesh = pyrender.Mesh.from_trimesh(original_mesh)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    # scene = pyrender.Scene()
    scene.add(mesh, "mesh")

    # add camera pose
    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]])
    # use this to make it to center
    camera = pyrender.camera.PerspectiveCamera(yfov=1)
    scene.add(camera, pose=camera_pose)

    # Get the lights from the viewer
    # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/3.0, outerConeAngle=np.pi/3.0)
    light = pyrender.SpotLight(
        color=100 * np.ones(3),
        intensity=1.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 3.0,
    )
    scene.add(light, pose=camera_pose)

    # offscreen render
    r = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(color[:, :, 0:3])
    # plt.show()
    # cv2.imwrite('obsvistest.png', color[:, :, 0:3])
    return color[:, :, 0:3]


def remove_samples_from_same_files(data, selected_idxes, num_samples: int = 3):
    new_selected_idxes = []
    source_files = []
    for idx in selected_idxes:
        if data["infos"][idx][0] in source_files:
            continue
        source_files.append(data["infos"][idx][0])
        new_selected_idxes.append(idx)
        if len(new_selected_idxes) >= num_samples:
            break
    return new_selected_idxes


def retrieve_and_save_imgs(
    data,
    test_data,
    batch_size: int = 10,
    num_samples: int = 3,
    key: str = "embeds",
    include_target: bool = False,
    idxes: np.ndarray = None,
):
    raw_inputs = data[key]
    if key == "observations" and include_target:
        raw_inputs = np.concatenate(
            [raw_inputs, data["future_observations"][:, -1]], axis=-1
        )
    normalized_raw_inputs = (
        raw_inputs / np.linalg.norm(raw_inputs, axis=1)[:, np.newaxis]
    )
    searcher = (
        scann.scann_ops_pybind.builder(normalized_raw_inputs, 20, "dot_product")
        .tree(
            num_leaves=2000,
            num_leaves_to_search=100,
            training_sample_size=20000,
        )
        .score_ah(2, anisotropic_quantization_threshold=0.2)
        .reorder(100)
        .build()
    )

    if idxes is None:
        selected_idxes = np.random.choice(
            test_data["observations"].shape[0], batch_size, replace=False
        )
    else:
        selected_idxes = idxes

    test_inputs = test_data[key]
    if key == "observations" and include_target:
        test_inputs = np.concatenate(
            [test_inputs, test_data["future_observations"][:, -1]], axis=-1
        )
    test_normalized_inputs = (
        test_inputs / np.linalg.norm(test_inputs, axis=1)[:, np.newaxis]
    )

    queries = [test_normalized_inputs[idx] for idx in selected_idxes]
    batch_knn_idxes = searcher.search_batched(queries)[0]

    filtered_batch_knn_idxes = []
    for idx in batch_knn_idxes:
        filtered_idx = remove_samples_from_same_files(data, idx, num_samples)
        filtered_batch_knn_idxes.append(filtered_idx)
    filtered_batch_knn_idxes = np.array(filtered_batch_knn_idxes)

    save_dir = "./images/{}{}".format(key, "-tar" if include_target else "")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for batch, knn_idxes in enumerate(filtered_batch_knn_idxes):
        obs = test_data["raw_observations"][selected_idxes[batch]]
        img = obs_visualize(obs)
        cv2.imwrite(save_dir + f"/{batch}.png", img)
        for fs in [4, 9, 14]:
            obs = test_data["raw_future_observations"][selected_idxes[batch]][fs]
            img = obs_visualize(obs)
            cv2.imwrite(save_dir + f"/{batch}--fs-{fs + 1}.png", img)

        for i, knn_idx in enumerate(knn_idxes[:3]):
            obs = data["raw_observations"][knn_idx]
            img = obs_visualize(obs)
            cv2.imwrite(save_dir + f"/{batch}-{i}--{knn_idx}.png", img)
            # for fs in range(len(data["raw_future_observations"][knn_idx])):
            for fs in [4, 9, 14]:
                obs = data["raw_future_observations"][knn_idx][fs]
                img = obs_visualize(obs)
                cv2.imwrite(save_dir + f"/{batch}-{i}--{knn_idx}--fs-{fs + 1}.png", img)

    return selected_idxes


if __name__ == "__main__":
    # data_path = "./logs/basic-retrieval-embedlr-0.001-embeddim-32-fstep-5/basic_retrieval-embedlr_0.001-embeddim_32-fstep_5_2022_09_27_11_26_16_0000--s-0/train_data-fs_15-embeds.th"
    # data = torch.load(data_path)
    # test_data_path = "./logs/basic-retrieval-embedlr-0.001-embeddim-32-fstep-5/basic_retrieval-embedlr_0.001-embeddim_32-fstep_5_2022_09_27_11_26_16_0000--s-0/test_data-fs_15-embeds.th"
    # test_data = torch.load(test_data_path)

    # selected_idxes = retrieve_and_save_imgs(data, test_data, key="embeds")
    # retrieve_and_save_imgs(data, test_data, key="raw_observations", idxes=selected_idxes)

    data_path = "./logs/basic-retrieval-cond-embedlr-0.001-embeddim-128-fstep-15/basic_retrieval-cond-embedlr_0.001-embeddim_128-fstep_15_2022_09_27_13_31_29_0000--s-0/train_data-fs_15-embeds.th"
    data = torch.load(data_path)
    test_data_path = "./logs/basic-retrieval-cond-embedlr-0.001-embeddim-128-fstep-15/basic_retrieval-cond-embedlr_0.001-embeddim_128-fstep_15_2022_09_27_13_31_29_0000--s-0/test_data-fs_15-embeds.th"
    test_data = torch.load(test_data_path)

    selected_idxes = retrieve_and_save_imgs(data, test_data, key="embeds", include_target=True)
    retrieve_and_save_imgs(data, test_data, key="observations", include_target=True, idxes=selected_idxes)
