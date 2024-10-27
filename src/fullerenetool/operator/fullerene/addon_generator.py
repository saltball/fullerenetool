try:
    import pynauty as pn
except ImportError:
    pn = None
    Warning(
        "pynauty not installed, cannot use addons_to_fullerene, ",
        "if you're using dflow, ignore this warning ",
        "if you install pynauty in your runtime environment.",
    )

from fullerenetool.fullerene.derivatives import DerivativeFullereneGraph
from fullerenetool.operator.graph import nx_to_nauty


def generate_addons_and_filter(
    fullerene_dev: DerivativeFullereneGraph,
    addon_num: int,
):
    """_summary_

    example:
    ```
    from ase.build import molecule
    C60 = molecule('C60')
    init_fullerene = FullereneCage(C60)
    dev_groups = [DerivativeGroup(
        atoms=ase.Atoms(
            'XCl',
            [
                [0.1, 0.1, -0.2],
                [0, 0, 0],
                # [0, 0, 1.4]
            ]
        ),
        graph=nx.from_numpy_array(
            np.array([[0, 1,], [1, 0,]])
        ),
        addon_atom_idx=0,
        # bond_length=1.4,
    )]*2
    dev_graph, dev_fullerenes = addons_to_fullerene(
        dev_groups,
        [0, 1],
        init_fullerene,
        nx.adjacency_matrix(init_fullerene.graph.graph).todense(),
    )
    devgraph = DerivativeFullereneGraph(
        adjacency_matrix=dev_graph.adjacency_matrix,
        cage_elements=dev_graph.node_elements,
        addons=[],
    )
    candidategraph_list = []
    for idx, candidategraph in enumerate(generate_addons_and_filter(devgraph, 1)):
        candidategraph_list.append(candidategraph)
        print(idx, candidategraph)
    ```

    Args:
        fullerene_dev (DerivativeFullereneGraph): _description_
        addon_num (int): _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    cage = fullerene_dev
    candidate_sites = fullerene_dev.addon_sites
    candidate_pairs = [[] for _ in range(addon_num)]
    candidate_graphs = [[] for _ in range(addon_num)]
    candidate_graph_certificate = [
        [] for _ in range(addon_num)
    ]  # nauty canonical label
    site_level = range(addon_num)
    candidate_pairs[0] = [list([site]) for site in candidate_sites]
    yield_flag = False
    for site_num in site_level:  # site 个数
        tmp_candidate_pairs = []  # 临时存储候选
        if site_num == addon_num - 1:
            yield_flag = True
        for isites in candidate_pairs[site_num]:  # 遍历生成的对
            isomorphic_flag = False
            tmp_cage_graph = cage.graph.copy()  # 操作的图
            for isite in isites:  # 遍历节点对中的节点
                new_node = len(tmp_cage_graph)  # 新的节点添加在末尾
                tmp_cage_graph.add_edge(new_node, isite)  # 新的图
            tmp_pn_graph = pn.canon_graph(
                nx_to_nauty(tmp_cage_graph, include_z_labels=False)
            )
            certif = pn.certificate(tmp_pn_graph)

            if certif in candidate_graph_certificate[site_num]:
                isomorphic_flag = True
                # continue
            if not isomorphic_flag:  # 如果不重复则加入列表
                candidate_graphs[site_num].append(tmp_cage_graph)
                candidate_graph_certificate[site_num].append(certif)
                if yield_flag:
                    yield isites, tmp_cage_graph
                tmp_candidate_pairs.append([*isites])
        candidate_pairs[site_num] = tmp_candidate_pairs  # 更新剔除重复后的候选
        # 更新新的对
        tmp_addons = []  #
        for candidate_pair in candidate_pairs[site_num]:  # 遍历剔除重复后的节点对
            for isite in candidate_sites:
                if isite not in candidate_pair:  # 仅考虑不在对中的新节点
                    tmp_addons.append([*candidate_pair, isite])
        if site_num < addon_num - 1:  # 更新新的候选遍历节点层级
            candidate_pairs[site_num + 1] = tmp_addons
    return candidate_pairs, candidate_graphs
