import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from fullerenetool.fullerene.cage import FullereneCage


def _create_projection_func(
    center_full,
    pro_radius,
    mat_rotate_3d,
    mat_rotate_2d,
    diameter_full,
    sphere_ratio,
    parr_ratio,
    center_circle,
    project_direct,
):
    def project_positions(positions):
        # 复制原始处理流程
        centered = positions - center_full
        pos_sphere = (
            centered / np.linalg.norm(centered, axis=1)[:, None] * diameter_full
        )

        # 半球投影和平行投影计算
        project_axis_s = hemisphere_projection_graph(
            pro_radius, pos_sphere, center_circle
        )
        project_axis_p = parrallel_projection_graph(
            pro_radius, pos_sphere, center_circle, project_direct
        )

        # 组合投影
        pro_t = (
            (-project_direct * project_axis_s).sum(-1)
            - pro_radius
            - (-project_direct * center_circle).sum(-1)
        )
        project_axis_sp = project_axis_s - pro_t[:, None] * -project_direct
        project_axis_sp = (
            sphere_ratio * project_axis_sp + parr_ratio * project_axis_p
        ) / 2

        # 应用旋转
        rotated_3d = np.einsum("an,mn->am", project_axis_sp, mat_rotate_3d)
        rotated_2d = rotated_3d.copy()
        rotated_2d[:, :2] = np.einsum("an,nm->am", mat_rotate_2d, rotated_3d[:, :2].T).T
        return rotated_2d

    return project_positions


def planarity_graph_pos(
    cage,
    sphere_ratio: float = 0.8,
    parr_ratio: float = 0.2,
    projection_point: int = 0,
    return_project_matrix: bool = False,
):
    """
    Draw fullerene planarity graph combining parallel and hemi-sphere projection.

    Parameters
    ----------
    cage: FullereneCage
        A planarizable graph of fullerene family.
    sphere_ratio, parr_ratio: float
        Ratios to control graph deformation between projection
        of platform and hemi-sphere.
    projection_point: int
        Index of the circle to use as the projection point.
        If set to None, it will use the geom-center of the first 6-member circle
        and average distance away from fullerene center.
    return_project_matrix: bool
        If True, return the projection matrix.

    Returns
    -------
    np.ndarray
        The projected matrix.
    np.ndarray
        The projection matrix. Only returned if return_project_matrix is True.
    """

    # Input validation
    if not isinstance(cage, FullereneCage):
        raise TypeError("cage must be an instance of FullereneCage")
    if not (0 <= sphere_ratio <= 1) or not (0 <= parr_ratio <= 1):
        raise ValueError("sphere_ratio and parr_ratio must be between 0 and 1")
    if not isinstance(projection_point, int) and projection_point is not None:
        raise TypeError("projection_point must be an integer or None")

    # Calculate molecular geom-center
    center_full = np.average(cage.positions, axis=0)
    centered_pos = cage.positions - center_full

    # Project on a sphere to avoid extreme deformation of shell
    diameter_full = np.max(np.linalg.norm(centered_pos, axis=1))
    pos_sphere = (
        centered_pos / np.linalg.norm(centered_pos, axis=1)[:, None] * diameter_full
    )

    # Select projection point
    circles = cage.circle_vertex_list
    projection_point_flag = 0
    circle_from = None

    for circle in circles:
        if len(circle) == 6:
            if projection_point >= 0 and projection_point_flag == projection_point:
                circle_from = circle
                break
            elif projection_point < 0 and projection_point_flag - 1 == projection_point:
                circle_from = circle
                break
            projection_point_flag += 1 if projection_point >= 0 else -1

    if circle_from is None:
        raise RuntimeError(
            "No suitable circle found for projection. Please check `projection_point`."
        )

    radius = np.average(np.linalg.norm(pos_sphere[circle_from], axis=1))

    # Calculate center of the selected circle
    center_circle = np.average(centered_pos[circle_from], axis=0)
    project_direct = center_circle / np.linalg.norm(center_circle)
    center_circle = project_direct * radius

    # Get the projection
    pro_radius = np.max(np.linalg.norm(pos_sphere - center_circle, axis=1))
    project_axis_s = hemisphere_projection_graph(
        pro_radius, pos_sphere, projection_from=center_circle
    )
    project_axis_p = parrallel_projection_graph(
        pro_radius,
        pos_sphere,
        projection_from=center_circle,
        project_direct_parrallel=project_direct,
    )

    # Projection from hemisphere to platform
    pro_t = (
        (-project_direct * project_axis_s).sum(-1)
        - pro_radius
        - (-project_direct * center_circle).sum(-1)
    )
    project_axis_sp = project_axis_s - pro_t[:, None] * -project_direct
    project_axis_sp = (sphere_ratio * project_axis_sp + parr_ratio * project_axis_p) / 2

    # Rotate to XoY
    mx, my, mz = project_direct
    axy = np.sqrt(mx * mx + my * my)
    mat_rotate_3d = np.array(
        [[mx * mz / axy, my * mz / axy, -axy], [-my / axy, mx / axy, 0], [mx, my, mz]]
    )
    project_axis_sp_r = np.einsum("an,mn->am", project_axis_sp, mat_rotate_3d)

    # Ensure a parallel top edge
    mx = project_axis_sp_r[:, 0]
    my = project_axis_sp_r[:, 1]
    mz = project_axis_sp_r[:, 2]
    sorted_my = np.sort(project_axis_sp_r[:, 1])
    originxyz1 = project_axis_sp_r[project_axis_sp_r[:, 1] == sorted_my[-1]]
    originxyz2 = project_axis_sp_r[project_axis_sp_r[:, 1] == sorted_my[-2]]
    topedge = originxyz2[:, :2] - originxyz1[:, :2]
    toporient = topedge / np.linalg.norm(topedge)
    topsin = float(toporient[:, 0])
    topcos = float(toporient[:, 1])
    mat_rotate_2d = np.array([[-topsin, -topcos], [topcos, -topsin]])
    project_axis_sp_r[:, :2] = np.einsum(
        "an,nm->am", mat_rotate_2d, np.array([mx, my])
    ).transpose()

    # Combine rotation matrices to form the final projection matrix
    projection_func = _create_projection_func(
        center_full,
        pro_radius,
        mat_rotate_3d,
        mat_rotate_2d,
        diameter_full,
        sphere_ratio,
        parr_ratio,
        center_circle,
        project_direct,
    )
    return project_axis_sp_r, projection_func


def planarity_graph_draw(
    cage,
    sphere_ratio: float = 0.8,
    parr_ratio: float = 0.2,
    projection_point: int = 0,
    path=None,
    pentage_color="orange",
    pentage_alpha=0.5,
    antialiased=True,
    line_color="orange",
    line_alpha=0.5,
    atom_label=True,
    ax=None,
):
    """
    Draw fullerene planarity graph combinating parrallel and hemi-sphere projection.

    Parameters
    ----------
    cage: FullereneCage
        A planaritable graph of fullerene family.
    sphere_ratio, parr_ratio:float
        ratio to control graph deformation between projection of platform
        and hemi-sphere.
    projection_point:str
        methods of choosing projection point
        If set to None, it will use the geom-center of the first 6-member circle and
        average distance away from fullerene center.
        # TODO: Group Point method.
    path:save to file
        if set to None, no file will be saved.

    Returns
    -------

    """
    circles = cage.circle_vertex_list
    project_axis_sp_r = planarity_graph_pos(
        cage,
        sphere_ratio=sphere_ratio,
        parr_ratio=parr_ratio,
        projection_point=projection_point,
    )

    # draw figure
    if ax is None:
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111)
        # ax.scatter(project_axis_p[:, 0], project_axis_p[:, 1], project_axis_p[:, 2])

    # draw circle and circle fills
    for circleone in circles:
        if len(circleone) == 5:
            xy = [
                project_axis_sp_r[circleone][:, 0],
                project_axis_sp_r[circleone][:, 1],
            ]
            ax.add_patch(
                mpatches.Polygon(
                    np.array(xy).transpose(),
                    color=pentage_color,
                    antialiased=antialiased,
                    alpha=pentage_alpha,
                )
            )

    # draw atoms
    if atom_label:
        ax.scatter(
            project_axis_sp_r[:, 0],
            project_axis_sp_r[:, 1],
            c=line_color,
            alpha=line_alpha,
            s=150,
        )
        for i in range(cage.natoms):
            ax.text(
                project_axis_sp_r[:, 0][i],
                project_axis_sp_r[:, 1][i],
                str(i + 1),
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
            )
            pass
        # draw edge lines
        for edges in cage.graph.edges:
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    project_axis_sp_r[edges[0]],
                    project_axis_sp_r[edges[1]],
                    antialiased=antialiased,
                    alpha=line_alpha,
                    shrinkA=7,
                    shrinkB=7,
                )
            )
    else:
        ax.scatter(
            project_axis_sp_r[:, 0],
            project_axis_sp_r[:, 1],
            c=line_color,
            alpha=line_alpha,
        )
        for edges in cage.graph.edges:
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    project_axis_sp_r[edges[0]],
                    project_axis_sp_r[edges[1]],
                    antialiased=antialiased,
                    alpha=line_alpha,
                    shrinkA=5,
                    shrinkB=5,
                )
            )

    # set the figure
    plt.axis("equal")
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])

    if not path:
        pass
    else:
        fig.savefig(path)
    return ax, project_axis_sp_r


def hemisphere_projection_graph(pro_radius, pos_sphere, projection_from):
    """
    Get projection axis using hemisphere method.
    Parameters
    ----------
    pro_radius:float
        projection length, from `projection_from` point to the projecting surface.
    pos_sphere:np.array
        axis of sphere projection, which can be safely projection using this method
        without cross lines.
    projection_from:np.array
        projection original point.
    Returns
    -------
    np.array
        hemisphere projection axis.
    """
    pro_vector = pos_sphere - projection_from
    pro_t = pro_radius / np.linalg.norm(pro_vector, axis=1)
    project_axis_s = projection_from + pro_t[:, None] * pro_vector
    return project_axis_s


def parrallel_projection_graph(
    pro_radius,
    pos_sphere,
    projection_from,
    project_direct_parrallel=np.array([0, 0, 0]),
):
    """
    Get projection axis using hemisphere method.
    Parameters
    ----------
    pro_radius:float
        projection length, from `projection_from` point to the projecting surface.
    pos_sphere:np.array
        axis of sphere projection, which can be safely projection using this method
        without cross lines.
    projection_from:np.array
        projection original point.
    project_direct_parrallel:np.array
        projection direction of this parrallel method, also the surface's normal vector
        (notice the difference as a "-").
    Returns
    -------
    np.array
        hemisphere projection axis.
    """
    pro_vector = pos_sphere - projection_from
    pro_t = pro_radius / (-project_direct_parrallel * pro_vector).sum(-1)
    project_axis_p = projection_from + pro_t[:, None] * pro_vector
    return project_axis_p
