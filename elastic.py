import numpy as np
from lammps import lammps


PRESSURE_COMPONENTS = ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]
BOX_COMPONENTS = ["lx", "ly", "lz"]


def extract_pressure(lmp):
    return {key: lmp.get_thermo(key) for key in PRESSURE_COMPONENTS}


def calculate_elastic_constants_lammps_api():
    """
    Calculates the elastic constant tensor for a crystal using the LAMMPS Python API,
    following the logic of the provided in.elastic, init.mod, potential.mod, and displace.mod files.
    """
    up = 1.0e-6
    atomjiggle = 1.0e-5
    units_style = "metal"
    cfac = 1.0e-4
    cunits = "GPa"
    etol = 0.0
    ftol = 1.0e-10
    maxiter = 100
    maxeval = 1000
    dmax = 1.0e-2
    a = 5.43
    lmp = lammps()
    lmp.commands_string(f"""
        variable up equal {up}
        variable atomjiggle equal {atomjiggle}
        units		{units_style}
        variable cfac equal {cfac}
        variable cunits string {cunits}
        variable etol equal {etol} 
        variable ftol equal {ftol}
        variable maxiter equal {maxiter}
        variable maxeval equal {maxeval}
        variable dmax equal {dmax}
        variable a equal {a}        
        boundary	p p p
        lattice         diamond $a
        region		box prism 0 2.0 0 3.0 0 4.0 0.0 0.0 0.0
        create_box	1 box
        create_atoms	1 box
        mass 1 1.0e-20
    """)
    lmp.commands_string(f"""
        pair_style	sw
        pair_coeff * * Si.sw Si
        neighbor 1.0 nsq
        neigh_modify once no every 1 delay 0 check yes
        min_style	     cg
        min_modify	     dmax ${{dmax}} line quadratic
        thermo		1
        thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
        thermo_modify norm no
    """)
    print("--- 1. Initial State Minimization ---")
    lmp.commands_string(f"""
        fix 3 all box/relax  aniso 0.0
        minimize ${{etol}} ${{ftol}} ${{maxiter}} ${{maxeval}}
        unfix 3
    """)
    pressure_0 = extract_pressure(lmp)
    print(pressure_0)
    box_dims = {key: lmp.get_thermo(key) for key in BOX_COMPONENTS}
    print(box_dims)
    lmp.command("write_restart restart.equil")
    lmp.command(
        f"displace_atoms all random {atomjiggle} {atomjiggle} {atomjiggle} 87287 units box"
    )
    C_neg = {i: np.zeros(6) for i in range(1, 7)}
    C_pos = {i: np.zeros(6) for i in range(1, 7)}
    for dir_val in range(1, 7):
        print(f"\n--- 2. Perturbation Cycle for dir={dir_val} ---")
        if dir_val == 1:
            len0 = box_dims["lx"]
        elif dir_val == 2:
            len0 = box_dims["ly"]
        elif dir_val == 3 or dir_val == 4 or dir_val == 5:
            len0 = box_dims["lz"]
        elif dir_val == 6:
            len0 = box_dims["ly"]
        xy = lmp.extract_box()[2]
        xz = lmp.extract_box()[3]
        yz = lmp.extract_box()[4]
        print(xy, xz, yz)
        lmp.commands_string("""
            clear
            box tilt large
            read_restart restart.equil
            include potential.mod
        """)
        print(f"  Negative deformation ($\epsilon_{{Voigt}} = -{up}$)")
        delta = -up * len0
        deltaxy = -up * xy
        deltaxz = -up * xz
        deltayz = -up * yz  # [cite: 5]
        if dir_val == 1:
            lmp.command(
                f"change_box all x delta 0 {delta} xy delta {deltaxy} xz delta {deltaxz} remap units box"
            )
        elif dir_val == 2:
            lmp.command(
                f"change_box all y delta 0 {delta} yz delta {deltayz} remap units box"
            )
        elif dir_val == 3:
            lmp.command(f"change_box all z delta 0 {delta} remap units box")
        elif dir_val == 4:
            lmp.command(f"change_box all yz delta {delta} remap units box")
        elif dir_val == 5:
            lmp.command(f"change_box all xz delta {delta} remap units box")
        elif dir_val == 6:
            lmp.command(f"change_box all xy delta {delta} remap units box")  # [cite: 6]
        lmp.command(f"minimize {etol} {ftol} {maxiter} {maxeval}")
        pressure_negative = extract_pressure(lmp)
        strain = delta / len0
        for i, key in enumerate(PRESSURE_COMPONENTS):
            C_neg[dir_val][i] = (
                -(pressure_negative[key] - pressure_0[key]) / strain * cfac
            )
        lmp.commands_string("""
            clear
            box tilt large
            read_restart restart.equil
            include potential.mod
        """)
        delta = up * len0
        deltaxy = up * xy
        deltaxz = up * xz
        deltayz = up * yz
        if dir_val == 1:
            lmp.command(
                f"change_box all x delta 0 {delta} xy delta {deltaxy} xz delta {deltaxz} remap units box"
            )  # [cite: 7]
        elif dir_val == 2:
            lmp.command(
                f"change_box all y delta 0 {delta} yz delta {deltayz} remap units box"
            )
        elif dir_val == 3:
            lmp.command(f"change_box all z delta 0 {delta} remap units box")
        elif dir_val == 4:
            lmp.command(f"change_box all yz delta {delta} remap units box")
        elif dir_val == 5:
            lmp.command(f"change_box all xz delta {delta} remap units box")
        elif dir_val == 6:
            lmp.command(f"change_box all xy delta {delta} remap units box")  # [cite: 8]
        lmp.command(f"minimize {etol} {ftol} {maxiter} {maxeval}")
        pressure_postive = extract_pressure(lmp)
        strain = delta / len0
        for i, key in enumerate(PRESSURE_COMPONENTS):
            C_pos[dir_val][i] = (
                -(pressure_postive[key] - pressure_0[key]) / strain * cfac
            )
    C_matrix = np.zeros((6, 6))
    for dir_val in range(1, 7):
        for i in range(6):
            C_matrix[i, dir_val - 1] = 0.5 * (C_neg[dir_val][i] + C_pos[dir_val][i])
    C_all = np.copy(C_matrix)
    C11all = C_all[0, 0]
    C22all = C_all[1, 1]
    C33all = C_all[2, 2]
    C44all = C_all[3, 3]
    C55all = C_all[4, 4]
    C66all = C_all[5, 5]
    C12all = 0.5 * (C_all[0, 1] + C_all[1, 0])
    C13all = 0.5 * (C_all[0, 2] + C_all[2, 0])
    C23all = 0.5 * (C_all[1, 2] + C_all[2, 1])
    C14all = 0.5 * (C_all[0, 3] + C_all[3, 0])
    C15all = 0.5 * (C_all[0, 4] + C_all[4, 0])
    C16all = 0.5 * (C_all[0, 5] + C_all[5, 0])
    C24all = 0.5 * (C_all[1, 3] + C_all[3, 1])
    C25all = 0.5 * (C_all[1, 4] + C_all[4, 1])
    C26all = 0.5 * (C_all[1, 5] + C_all[5, 1])
    C34all = 0.5 * (C_all[2, 3] + C_all[3, 2])
    C35all = 0.5 * (C_all[2, 4] + C_all[4, 2])
    C36all = 0.5 * (C_all[2, 5] + C_all[5, 2])
    C45all = 0.5 * (C_all[3, 4] + C_all[4, 3])
    C46all = 0.5 * (C_all[3, 5] + C_all[5, 3])
    C56all = 0.5 * (C_all[4, 5] + C_all[5, 4])
    C11cubic = (C11all + C22all + C33all) / 3.0
    C12cubic = (C12all + C13all + C23all) / 3.0
    C44cubic = (C44all + C55all + C66all) / 3.0
    bulkmodulus = (C11cubic + 2 * C12cubic) / 3.0
    shearmodulus1 = C44cubic
    shearmodulus2 = (C11cubic - C12cubic) / 2.0
    poissonratio = 1.0 / (1.0 + C11cubic / C12cubic)
    print("\n=========================================")
    print("Components of the Elastic Constant Tensor")
    print("=========================================")
    print(f"Elastic Constant C11all = {C11all:.4f} {cunits}")
    print(f"Elastic Constant C22all = {C22all:.4f} {cunits}")
    print(f"Elastic Constant C33all = {C33all:.4f} {cunits}")
    print(f"Elastic Constant C12all = {C12all:.4f} {cunits}")
    print(f"Elastic Constant C13all = {C13all:.4f} {cunits}")
    print(f"Elastic Constant C23all = {C23all:.4f} {cunits}")
    print(f"Elastic Constant C44all = {C44all:.4f} {cunits}")
    print(f"Elastic Constant C55all = {C55all:.4f} {cunits}")
    print(f"Elastic Constant C66all = {C66all:.4f} {cunits}")
    print("\n=========================================")
    print("Average properties for a cubic crystal")
    print("=========================================")
    print(f"Bulk Modulus = {bulkmodulus:.4f} {cunits}")
    print(f"Shear Modulus 1 = {shearmodulus1:.4f} {cunits}")
    print(f"Shear Modulus 2 = {shearmodulus2:.4f} {cunits}")
    print(f"Poisson Ratio = {poissonratio:.4f}")
    print(
        "\n(Note: For Stillinger-Weber silicon, analytical results are C11=151.4 GPa, C12=76.4 GPa, C44=56.4 GPa) [cite: 27]"
    )
    lmp.close()
    return C_all


if __name__ == "__main__":
    final_C_matrix = calculate_elastic_constants_lammps_api()
    print(final_C_matrix)
