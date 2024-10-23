import time
from pathlib import Path
from typing import List

import networkx as nx
from dflow import (
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    argo_len,
    argo_sequence,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    BigParameter,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)

from fullerenetool.fullerene.derivatives import (
    DerivativeFullereneGraph,
    DerivativeGroup,
    FullereneCage,
    addons_to_fullerene,
)
from fullerenetool.logger import logger
from fullerenetool.operator.fullerene.addon_generator import generate_addons_and_filter


class GetNonisomorphicAddons(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "fulleren_init": BigParameter(FullereneCage),
                "addon": BigParameter(DerivativeGroup),
                "addon_start": int,
                "start_idx": List[int],
                "add_num": int,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "candidategraph_list_file": Artifact(Path),
                "candidategraph_name_list_file": Artifact(Path),
                "addon_pos_index_list_file": Artifact(Path),
                "atoms_file_list": Artifact(List[Path]),
                "task_num": int,
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        import jsonpickle
        from ase.io.extxyz import write_extxyz

        fulleren_init = op_in["fulleren_init"]
        addon = op_in["addon"]
        addon_start = op_in["addon_start"]
        start_idx = op_in["start_idx"]
        add_num = op_in["add_num"]

        dev_groups = [addon] * addon_start
        dev_graph, dev_fullerenes = addons_to_fullerene(
            dev_groups,
            start_idx,
            fulleren_init,
            nx.adjacency_matrix(fulleren_init.graph.graph).todense(),
        )
        devgraph = DerivativeFullereneGraph(
            adjacency_matrix=dev_graph.adjacency_matrix,
            cage_elements=dev_graph.node_elements,
            addons=[],
        )
        origin_dev = devgraph.generate_atoms_with_addons(
            b2n2_steps=50, traj="traj", init_pos=dev_fullerenes.positions
        )
        candidategraph_list = []
        candidategraph_name_list = []
        atoms_list = []
        addon_pos_index_list = []

        for idx, candidategraph in enumerate(
            generate_addons_and_filter(devgraph, add_num)
        ):
            st_time = time.perf_counter()
            logger.info("{} {}".format(idx, candidategraph))
            graph = candidategraph[1]
            dev_fullerene = DerivativeFullereneGraph(
                adjacency_matrix=nx.adjacency_matrix(graph).todense(),
                cage_elements=devgraph.node_elements,
                addons=[addon] * add_num,
            ).generate_atoms_with_addons(
                algorithm="cagethenaddons",
                init_pos=origin_dev.positions,
                check=False,
                use_gpu=True,
            )
            fl_time = time.perf_counter() - st_time
            logger.info(f"run time: {fl_time:.6f} s")
            candidategraph_list.append(list(int(i) for i in candidategraph[0]))
            new_isomer_name = "{}_{}({})_add({})".format(
                fulleren_init.name,
                addon.name,
                "_".join(str(start_idx)) if start_idx else "_none",
                "_".join(str(i) for i in candidategraph[0]),
            )
            addon_pos_index = "_".join(
                [
                    str(index)
                    for index in sorted([*start_idx, *[i for i in candidategraph[0]]])
                ]
            )
            candidategraph_name_list.append(new_isomer_name)
            write_extxyz(Path(new_isomer_name + ".xyz").open("w"), dev_fullerene)
            atoms_list.append(Path(new_isomer_name + ".xyz"))
            addon_pos_index_list.append(addon_pos_index)

        Path("candidategraph_list_file").write_text(
            "\n".join(
                ",".join(str(i) for i in graph_index)
                for graph_index in candidategraph_list
            )
        )
        Path("candidategraph_name_list_file").write_text(
            "\n".join(candidategraph_name_list)
        )
        Path("addon_pos_index_list_file").write_text(
            jsonpickle.dumps(addon_pos_index_list)
        )

        op_out = OPIO(
            {
                "candidategraph_list_file": Path("candidategraph_list_file"),
                "candidategraph_name_list_file": Path("candidategraph_name_list_file"),
                "addon_pos_index_list_file": Path("addon_pos_index_list_file"),
                "atoms_file_list": atoms_list,
                "task_num": len(candidategraph_list),
            }
        )

        return op_out


class GatherNonisomorphicAddons(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "candidategraph_list_file": Artifact(Path),
                "candidategraph_name_list_file": Artifact(Path),
                "addon_pos_index_list_file": Artifact(Path),
                "atoms_file_list": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign(
            {
                "candidategraph_list": List[List[int]],
                "candidategraph_name_list": BigParameter(List[str]),
                "atoms_file_list": Artifact(List[Path]),
                "task_num": int,
            }
        )

    @OP.exec_sign_check
    def execute(self, op_in):
        import jsonpickle

        addon_pos_index_list = jsonpickle.loads(
            op_in["addon_pos_index_list_file"].read_text()
        )
        op_in_candidategraph_list = [
            list(map(int, i.split(",")))
            for i in op_in["candidategraph_list_file"].read_text().split("\n")
        ]
        op_in_candidategraph_name_list = (
            op_in["candidategraph_name_list_file"].read_text().split("\n")
        )
        label_list = []
        candidategraph_list = []
        candidategraph_name_list = []
        atoms_list = []
        for i, _ in enumerate(op_in_candidategraph_name_list):
            label = addon_pos_index_list[i]
            if label not in label_list:
                label_list.append(label)
                candidategraph_list.append(op_in_candidategraph_list[i])
                candidategraph_name_list.append(op_in_candidategraph_name_list[i])
                atoms_list.append(op_in["atoms_file_list"][i])

        op_out = OPIO(
            {
                "candidategraph_list": candidategraph_list,
                "candidategraph_name_list": candidategraph_name_list,
                "atoms_file_list": atoms_list,
                "task_num": len(candidategraph_list),
            }
        )
        return op_out


def establish_parrel_generate_nonisomorphic_addon_steps(
    simple_machine_template_config,
    gpu_machine_template_config,
    group_size,
    image: str,
    generate_addons_group_size=1,
):
    steps = Steps(
        name="generate-nonisomorphic-addon-steps",
        inputs=Inputs(
            parameters={
                "fulleren_init": InputParameter(),
                "addon": InputParameter(),
                "addon_start": InputParameter(),
                "start_idx_list": InputParameter(),
                "add_num": InputParameter(),
            }
        ),
        outputs=Outputs(
            parameters={
                "candidategraph_list": OutputParameter(),
                "candidategraph_name_list": OutputParameter(),
                "task_num": OutputParameter(),
            },
        ),
    )
    generate_nonisomorphic_addon_step = Step(
        name="generate-nonisomorphic-addon",
        template=PythonOPTemplate(
            GetNonisomorphicAddons,
            image=image,
            **gpu_machine_template_config,
        ),
        slices=Slices(
            input_parameter=["start_idx"],
            output_artifact=[
                "candidategraph_list_file",
                "candidategraph_name_list_file",
                "atoms_file_list",
                "addon_pos_index_list_file",
            ],
            group_size=generate_addons_group_size,
            pool_size=1,
        ),
        parameters={
            "fulleren_init": steps.inputs.parameters["fulleren_init"],
            "addon": steps.inputs.parameters["addon"],
            "addon_start": steps.inputs.parameters["addon_start"],
            "start_idx": steps.inputs.parameters["start_idx_list"],
            "add_num": steps.inputs.parameters["add_num"],
        },
        with_sequence=argo_sequence(
            argo_len(steps.inputs.parameters["start_idx_list"])
        ),
    )
    steps.add(generate_nonisomorphic_addon_step)
    gather_nonisomorphic_addon_step = Step(
        name="gather-nonisomorphic-addon",
        template=PythonOPTemplate(
            GatherNonisomorphicAddons,
            image=image,
            **simple_machine_template_config,
        ),
        artifacts={
            "candidategraph_list_file": (
                generate_nonisomorphic_addon_step.outputs.artifacts[
                    "candidategraph_list_file"
                ]
            ),
            "candidategraph_name_list_file": (
                generate_nonisomorphic_addon_step.outputs.artifacts[
                    "candidategraph_name_list_file"
                ]
            ),
            "atoms_file_list": generate_nonisomorphic_addon_step.outputs.artifacts[
                "atoms_file_list"
            ],
            "addon_pos_index_list_file": (
                generate_nonisomorphic_addon_step.outputs.artifacts[
                    "addon_pos_index_list_file"
                ]
            ),
        },
    )
    steps.add(gather_nonisomorphic_addon_step)
    steps.outputs.parameters["candidategraph_list"].value_from_parameter = (
        gather_nonisomorphic_addon_step.outputs.parameters["candidategraph_list"]
    )

    steps.outputs.parameters["candidategraph_name_list"].value_from_parameter = (
        gather_nonisomorphic_addon_step.outputs.parameters["candidategraph_name_list"]
    )

    steps.outputs.artifacts["atoms_file_list"] = OutputArtifact(
        _from=gather_nonisomorphic_addon_step.outputs.artifacts["atoms_file_list"]
    )
    steps.outputs.parameters["task_num"].value_from_parameter = (
        gather_nonisomorphic_addon_step.outputs.parameters["task_num"]
    )
    return steps
