import argparse
import os
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm


def parse_lammps_dump(filename):
    num_steps = []
    num_atoms = []
    box_bounds = []

    with open(filename, "r") as file:
        while True:
            line = file.readline()
            if not line:
                break  # EOF

            if line.startswith("ITEM: TIMESTEP"):
                num_steps.append(int(file.readline().strip()))
                continue

            if line.startswith("ITEM: NUMBER OF ATOMS"):
                num_atoms.append(int(file.readline().strip()))

            if line.startswith("ITEM: BOX BOUNDS"):
                box_bounds_item = []
                for _ in range(3):
                    box_bounds_item.append([float(x) for x in file.readline().split()])
                box_bounds.append(box_bounds_item)

    if np.unique(num_atoms).size == 1:
        num_atoms = num_atoms[0]
    else:
        num_atoms = list(np.unique(num_atoms))
    if np.unique(box_bounds).size == 1:
        box_bounds = box_bounds[0].to_list()
    else:
        box_bounds = np.unique(box_bounds, axis=0).tolist()

    step_interval = np.diff(num_steps)
    if len(step_interval) == 1:
        step_interval = step_interval[0]
    else:
        step_interval = list(np.unique(step_interval))
    return {
        "num_steps": num_steps,
        "step_interval": step_interval,
        "num_atoms": num_atoms,
        "box_bounds": box_bounds,
    }


def read_lammps_dump(filename, slice_: slice):
    data = {}
    steps = []
    box_bounds = []
    atom_format = []

    tbar = tqdm()
    start_step = slice_.start
    stop_step = slice_.stop
    add_flag = False

    with open(filename, "r") as file:
        while True:
            line = file.readline()
            if not line:
                break  # EOF

            if line.startswith("ITEM: TIMESTEP"):
                tbar.update(1)
                step = int(file.readline().strip())
                if (
                    (step - start_step) % slice_.step != 0
                    or step > stop_step
                    or step < start_step
                ):
                    add_flag = False
                    continue
                else:
                    add_flag = True
                steps.append(step)

            if not add_flag:
                continue

            if line.startswith("ITEM: NUMBER OF ATOMS"):
                num_atoms = int(file.readline().strip())

            if line.startswith("ITEM: BOX BOUNDS"):
                box_bounds_item = []
                for _ in range(3):
                    box_bounds_item.append([float(x) for x in file.readline().split()])
                box_bounds.append(box_bounds_item)

            if line.startswith("ITEM: ATOMS"):
                atom_data_item = []
                atom_format.append(line.split("ITEM: ATOMS ")[1])
                for _ in range(num_atoms):
                    atom_data_item.append([x for x in file.readline().split()])
                data[step] = atom_data_item

    filtered_steps = data.keys()

    return {
        "data": data,
        "steps": filtered_steps,
        "box_bounds": box_bounds,
        "num_steps": len(steps),
        "atom_format": atom_format,
    }


def write_hdf5(lammps_dump_data, output_filename, steps):
    with h5py.File(output_filename, "w") as f:

        group = f.create_group("trajectory")
        for idx, step in enumerate(steps):
            # Write data to HDF5 file
            dset = group.create_dataset(
                str(step),
                data="\n".join(
                    " ".join(atom_line) for atom_line in lammps_dump_data["data"][step]
                ),
            )
            # Add timestep attribute
            dset.attrs["timestep"] = step
            dset.attrs["box_bounds"] = lammps_dump_data["box_bounds"][idx]
            dset.attrs["atom_format"] = lammps_dump_data["atom_format"]

        f.attrs["format_version"] = 1.0
        f.attrs["creation_date"] = str(datetime.now())


def decompress_hdf5_to_dump(hdf5_filename, output_filename):
    with h5py.File(hdf5_filename, "r") as f:
        traj_group = f["trajectory"]
        with open(output_filename, "w") as out_file:
            for key in sorted(traj_group.keys(), key=int):
                step = traj_group[key].attrs["timestep"]
                atoms = str(np.array(traj_group[key], dtype=str))

                out_file.write(f"ITEM: TIMESTEP\n{step}\n")
                atom_len = len(atoms.split("\n"))
                out_file.write(f"ITEM: NUMBER OF ATOMS\n{atom_len}\n")
                out_file.write("ITEM: BOX BOUNDS pp pp pp\n")
                bound = traj_group[key].attrs["box_bounds"]
                for i in range(3):
                    out_file.write(f"{bound[i][0]} {bound[i][1]}\n")
                out_file.write("ITEM: ATOMS " + traj_group[key].attrs["atom_format"][0])
                out_file.write(atoms + "\n")


def main():
    parser = argparse.ArgumentParser(description="Process LAMMPS dump files.")
    subparsers = parser.add_subparsers(dest="command")

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Display information about the dump file."
    )
    info_parser.add_argument(
        "filename", type=str, help="The LAMMPS dump file to inspect."
    )

    # Compress command
    compress_parser = subparsers.add_parser(
        "compress", help="Compress the dump file to HDF5 format."
    )
    compress_parser.add_argument(
        "filename", type=str, help="The LAMMPS dump file to compress."
    )
    compress_parser.add_argument(
        "--slice",
        type=lambda s: slice(
            *map(int, (s.replace(" ", "").replace("[", "").replace("]", "").split(",")))
        ),
        required=True,
        help="The slice of time steps to include, e.g., [0,1000,2] or [0,-1,100].",
    )

    # Decompress command
    decompress_parser = subparsers.add_parser(
        "decompress", help="Decompress an HDF5 file back to LAMMPS dump format."
    )
    decompress_parser.add_argument(
        "filename", type=str, help="The HDF5 file to decompress."
    )
    decompress_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="decompressed.dump",
        help="Output LAMMPS dump file name.",
    )

    args = parser.parse_args()

    if args.command == "info":
        if not os.path.exists(args.filename):
            print(f"Error: File '{args.filename}' does not exist.")
            exit(1)
            return
        lammps_info = parse_lammps_dump(args.filename)
        print(f"Number of timesteps: {len(lammps_info['num_steps'])}")
        print(
            f"""From step {lammps_info['num_steps'][0]} to {
                lammps_info['num_steps'][-1]}"""
        )
        print(f"Step Interval: {lammps_info['step_interval']}")
        print(f"Number of atoms: {lammps_info['num_atoms']}")
        print(f"Box bounds: {lammps_info['box_bounds']}")

    elif args.command == "compress":
        lammps_dump_data = read_lammps_dump(args.filename, args.slice)
        output_filename = f"{args.filename}.h5"
        try:
            write_hdf5(lammps_dump_data, output_filename, lammps_dump_data["steps"])
        except KeyError as e:
            print(
                f"""Make sure the steps are within the range of the dump file.
                    Use `lammps_slice info {args.filename} ` to check the range."""
            )
            raise e
        print(f"Compression completed. Saved to {output_filename}")

    elif args.command == "decompress":
        decompress_hdf5_to_dump(args.filename, args.output)
        print(f"Decompression completed. Saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    # import os
    # from pathlib import Path
    # os.chdir('/root/WORK/fullerenetool/.local//tool_tests')
    # decompress_hdf5_to_dump(
    #     'growth.dump.h5', 'lammps_slice_output.dump'
    # )
