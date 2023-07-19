import numpy as np


def get_memory_energy(self, system):
    sz_list = self.get_sz_list(system)
    loc_list = self.get_loc_list()
    memorgy_energy = 0
    for tensor_sz, loc in zip(sz_list, loc_list):
        if loc == "off":
            energy = system.energy_per_offchip_access + system.energy_per_onchip_access
        elif loc == "on":
            energy = system.energy_per_onchip_access
        else:
            raise ValueError(f"Wrong bw allocation: {loc}.")
        memorgy_energy += tensor_sz * energy
    return memorgy_energy


def get_noc_energy(self, system):
    sz_list = self.get_sz_list(system)
    loc_list = self.get_loc_list()
    noc_energy = 0
    for tensor_sz, loc in zip(sz_list, loc_list):
        if loc == "off":
            energy = (
                system.energy_per_data_byte_offchip_to_onchip
                + system.energy_per_data_byte_onchip_to_compute
            )
        elif loc == "on":
            energy = system.energy_per_data_byte_onchip_to_compute
        else:
            raise ValueError(f"Wrong bw allocation: {loc}.")
        noc_energy += tensor_sz * energy
    return noc_energy


## A -> M*K .   B -> K*N .   C-> M*N
## Internal Dataflow

## We have 2 levels of compute heirarchy
#
# [H,W, C1,C2]
#
# There are HxW compute cores and each core has C1xC2 computing elements.
## Total amount of work to be done is M*N*K Computations
""" Loop Order

Just compute:
Inner product:
    for M
        for N
            for K

Distributed inner product:
    for Tm
        for Tn
            for Tk
                for m
                    for n
                        for k

"""


def get_a_stationary_access(M, N, K, compute, self, system):
    if compute[0] == compute[1]:
        c1 = compute[0]
        c2 = c1
    else:
        c1 = compute[0]
        c2 = compute[1]

    mat_densities = self.get_density_list()
    pe_sparsity = system.pe_min_density_support
    num_loop = np.ceil(min(M, K) / c2) * np.ceil(max(M, K) / c1)

    a_reads_per_loop = np.prod(compute) * mat_densities[0]
    b_reads_per_loop = c1 * N * mat_densities[1]
    c_reads_write_per_loop = 2 * c2 * N * mat_densities[2]

    total_access = num_loop * (
        a_reads_per_loop + b_reads_per_loop + c_reads_write_per_loop
    )

    # ## Required SRAM Amount
    # sram_required = a_reads_per_loop + 2

    ## N dimension is streamming         ## Num of computations reduce proportional to matrix densities
    total_compute_time = num_loop * N * max(np.prod(mat_densities), pe_sparsity)

    return total_access, total_compute_time


def get_b_stationary_access(M, N, K, compute, self, system):
    if compute[0] == compute[1]:
        c1 = compute[0]
        c2 = c1
    else:
        c1 = compute[0]
        c2 = compute[1]

    mat_densities = self.get_density_list()
    pe_sparsity = system.pe_min_density_support
    num_loop = np.ceil(min(N, K) / c1) * np.ceil(max(N, K) / c2)

    a_reads_per_loop = M * c2 * mat_densities[0]
    b_reads_per_loop = np.prod(compute) * mat_densities[1]
    c_reads_write_per_loop = 2 * c1 * M * mat_densities[2]

    total_access = num_loop * (
        a_reads_per_loop + b_reads_per_loop + c_reads_write_per_loop
    )

    ## M dimension is streamming        ## Num of computations reduce proportional to matrix densities
    total_compute_time = num_loop * M * max(np.prod(mat_densities), pe_sparsity)
    return total_access, total_compute_time


def get_c_stationary_access(M, N, K, compute, self, system):
    if compute[0] == compute[1]:
        c1 = compute[0]
        c2 = c1
    else:
        c1 = compute[0]
        c2 = compute[1]

    mat_densities = self.get_density_list()
    pe_sparsity = system.pe_min_density_support
    num_loop = np.ceil(M / c2) * np.ceil(N / c1)

    a_reads_per_loop = c2 * K * mat_densities[0]
    b_reads_per_loop = c1 * K * mat_densities[1]
    c_reads_write_per_loop = 2 * np.prod(compute) * mat_densities[2]

    total_access = num_loop * (
        a_reads_per_loop + b_reads_per_loop + c_reads_write_per_loop
    )

    ## K dimension is streamming        ## Num of computations reduce proportional to matrix densities
    total_compute_time = num_loop * K * max(np.prod(mat_densities), pe_sparsity)

    return total_access, total_compute_time


def per_core_energy_perforamce(M, N, K, per_core_compute, system, self):
    per_access_energy = system.energy_per_data_byte_onchip_to_compute
    per_compute_energy = system.energy_per_mac

    ## This is for each smaller compute DF
    a_stationary_access, a_stationary_compute = get_a_stationary_access(
        M, N, K, per_core_compute, self, system
    )
    b_stationary_access, b_stationary_compute = get_b_stationary_access(
        M, N, K, per_core_compute, self, system
    )
    c_stationary_access, c_stationary_compute = get_c_stationary_access(
        M, N, K, per_core_compute, self, system
    )

    ## Energy = Element reads *read energy  +    compute time * num PEs *read energy
    total_energy_cost_a = (
        a_stationary_access * per_access_energy
        + (a_stationary_compute * np.prod(per_core_compute)) * per_compute_energy
    )
    total_energy_cost_b = (
        b_stationary_access * per_access_energy
        + (b_stationary_compute * np.prod(per_core_compute)) * per_compute_energy
    )
    total_energy_cost_c = (
        c_stationary_access * per_access_energy
        + (c_stationary_compute * np.prod(per_core_compute)) * per_compute_energy
    )

    energies = np.array([total_energy_cost_a, total_energy_cost_b, total_energy_cost_c])
    performances = np.array(
        [a_stationary_compute, b_stationary_compute, c_stationary_compute]
    )

    return energies, performances


def get_matmul_access(self, system):
    per_sram_change_energy = system.energy_per_data_byte_core_to_core
    per_dram_change_energy = system.energy_per_data_byte_offchip_to_onchip

    SRAM = system.on_chip_mem_size

    if system.mxu_shape is not None:
        compute = system.mxu_shape
    else:
        num_pes = np.sqrt(system.op_per_sec)
        compute = [1, 1, num_pes, num_pes]

    N, M, K, B = self.get_gemms()
    sz_list = self.get_sz_list(system)

    A_size = sz_list[0]
    B_size = sz_list[1]
    C_size = sz_list[2]

    ## This is for each smaller compute DF
    per_core_compute = compute[-2:]
    num_rows = min(compute[0], compute[1])
    num_cols = max(compute[0], compute[1])
    # assert(num_cols == num_rows)

    # number_of access and time per core.
    energies, performances = per_core_energy_perforamce(
        M // num_cols, N // num_cols, K // num_rows, per_core_compute, system, self
    )

    ## Based on on-chip memory implications
    if A_size + 2 * B_size + C_size < SRAM:
        on_chip_memory_access = B_size
    else:
        SRAM /= 2  ## /2 for double buffering
        on_chip_memory_access = (
            min(
                np.ceil(min(A_size, B_size) / SRAM) * max(A_size, B_size)
                + min(A_size, B_size),
                np.ceil(max(A_size, B_size) / SRAM) * min(A_size, B_size)
                + max(A_size, B_size),
            )
            + C_size
        )

    dram_energy = on_chip_memory_access * per_dram_change_energy

    ## We stream the 2 matrix which has the smallest footprint. We keep the largest matrix stationary.
    between_core_movements = (
        min(A_size + B_size, A_size + C_size, B_size + C_size)
        * (num_rows - 1)
        * per_sram_change_energy
    )

    ## Each iteration there are num_rows^2 cores and we do num_row such iteration
    total_core_energy = energies * np.power(num_cols, 2) * num_rows

    # print(total_core_energy,between_core_movements,dram_energy)
    total_energy = total_core_energy + between_core_movements + dram_energy

    performances = (performances * num_cols) * B
    energy_performance = total_energy * performances

    optimal_PP = np.min(energy_performance)

    return optimal_PP, np.min(performances), performances, total_energy
