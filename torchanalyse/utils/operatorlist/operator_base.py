import numpy as np
from operator import mul
from math import ceil
from .energy_cost import *
import warnings
op_type_dicts = {0: 'FC', 1: 'CONV2D', 2: 'DWCONV', 3: 'GEMM', 4: 'Logit', 5: 'Attend'}
class Operator(object):
    def __init__(self,node, density=(1.0,1.0,1.0)):
        self.node = node
        self.density_a, self.density_w, self.density_o = density
        tensors = self.get_tensors()
        if len(tensors) == 3:
            self.input_a = tensors[0]
            self.input_w = tensors[1]
            self.output = tensors[2]
        elif len(tensors) == 2:
            self.input_a = tensors[0]
            self.input_w = None
            self.output = tensors[1]
        else:
            warnings.warn("tensor size not imported:{}".format(node.operator))
        self.num_ops = self.get_num_ops()
        self.set_mem_pin(*self.get_default_mem_loc())

    

    def get_default_mem_loc(self):
        return ['off', 'off', 'off']

    def set_mem_pin(self, input_a=None, input_w=None, output=None):
        if input_a is not None:
            self.input_a_loc = input_a
        if input_w is not None:
            self.input_w_loc = input_w
        if output is not None:
            self.output_loc = output

    def set_tensor(self, input_a=None, input_w=None, output=None):
        if input_a is not None:
            self.input_a = input_a
        if input_w is not None:
            self.input_w = input_w
        if output is not None:
            self.output = output

    def get_density_list(self):
        return [self.density_a, self.density_w, self.density_o]

    def get_op_type(self, dim):
        return op_type_dicts[dim[-1]]

    def get_tensors(self):
        pass

    def get_size(self, tensor):
        return np.prod(tensor)

    # Each kind of operation function will have its own num ops, in which using the layer parameters obtained from the 
    # .csv file it will give out number of required ops .
    def get_num_ops(self):
        pass

    # For each kind of operator, this returns number of required paramters for that layer type. (Refer operators.py )
    def get_effective_dim_len(self):
        pass

    def get_num_data(self):
        return sum(self.get_sz_list())

    def get_effective_num_data(self, system):
        return sum(self.get_sz_list(system))

    # Total num ops for particular layer / number of system's ops . 
    def get_ideal_compute_time(self, system):
        return self.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec

    # Total number of memory paramters of layer / mem_BW.
    def get_ideal_memory_time(self, system):
        sz_list = self.get_sz_list(system)
        # print (sz_list)
        # print (system.get_bit_multiplier(type='M'))
        # print (system.offchip_mem_bw)
        # print (memory_time_offchip)
        memory_time_onchip = 0
        memory_time_offchip = 0
        for tensor_sz in sz_list:
            memory_time_onchip += tensor_sz * system.get_bit_multiplier(type='M')/ system.onchip_mem_bw
            memory_time_offchip += tensor_sz * system.get_bit_multiplier(type='M')/ system.offchip_mem_bw
        return  memory_time_offchip, memory_time_onchip



    def get_compute_efficiency(self, mxu_shape, mxu_mapping):
        outer_iters = ceil(mxu_mapping[0]/mxu_shape[0])
        inner_iters = []
        for mxu_size, dim_size in zip(mxu_shape[1:], mxu_mapping[1:]):
            inner_iter_cur = ceil(dim_size/mxu_size)
            inner_iters.append(inner_iter_cur)
        iters = [outer_iters] + inner_iters
        num_iters = np.prod(iters)
        efficiency = np.prod(mxu_mapping) / (num_iters * np.prod(mxu_shape))
        return num_iters, efficiency



    def get_effective_mxu_mapping(self, system):
        left, upper, contract, outer = self.get_gemms()
        # print("First:",left, upper, contract, outer)
        if system.skip_compute:
            contract = contract * self.density_w * self.density_a
            if contract < 1:
                print(f'[Warning] Contract dimension < 1, after sparsified')
            if system.skip_compute_on_noopt_output:
                # print("inside skip compute loop", left, self.density_o)
                left = left *self.density_o
        # print("Sec:",left, upper, contract, outer)
        mxu_mapping = np.sort([left, upper, contract])[::-1]
        # print("mxu_mapping:",mxu_mapping)
        *mxu_mapping, streaming_dim = [outer] + [m for m in mxu_mapping]
        return mxu_mapping, streaming_dim

    def get_effective_mxu_shape(self, mxu_shape):
        effective_mxu_shape = np.sort(mxu_shape[-2:])[::-1]
        effective_mxu_shape = [1] + [m for m in effective_mxu_shape]
        # effective_mxu_shape = [mxu_shape[0]] + [m for m in effective_mxu_shape]
        return effective_mxu_shape

    ##TODO: What is refered here as mxu shape, without that efficiency is 1 now.
    ## This module should take into account the system parameters and give effective run time and efficancy
    def get_compute_time(self, system):
        if system.mxu_shape is not None:
            mxu_mapping, _ = self.get_effective_mxu_mapping(system)
            effective_mxu_shape = self.get_effective_mxu_shape(system.mxu_shape)
            # print(effective_mxu_shape, mxu_mapping)
            _ , compute_efficiency = self.get_compute_efficiency(effective_mxu_shape, mxu_mapping)
            # print(compute_efficiency)
        elif(system.accelerator_type=="unstructured" and (self.density_a*self.density_w*self.density_o < 1) and system.treat_as_dense == False):
            # print("Puting inefficiency due to unstructured")
            compute_efficiency = system.unstructured_efficiency
        else:
            compute_efficiency = 1
        compute_efficiency = min(1,compute_efficiency)      ## Max efficiency is 1.0
        return self.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec / compute_efficiency, compute_efficiency

    def get_mxu_energy(self, system):
        if system.mxu_shape is not None:
            mxu_mapping, streaming_dim = self.get_effective_mxu_mapping(system)
            power_gating_mxu_shape = [ m//g for m, g in zip(system.mxu_shape, system.power_gating_granularity)]
            effective_mxu_shape = self.get_effective_mxu_shape(power_gating_mxu_shape)
            pg_num_iters, compute_efficiency = self.get_compute_efficiency(effective_mxu_shape, mxu_mapping)
            effective_mxu_shape = self.get_effective_mxu_shape(system.mxu_shape)
            origin_num_iters, compute_efficiency = self.get_compute_efficiency(effective_mxu_shape, mxu_mapping)
            energy_per_power_gated_mxu = system.energy_per_4_128_128_mxu / np.prod(system.power_gating_granularity)
            power_gated_energy = energy_per_power_gated_mxu * pg_num_iters * streaming_dim
            energy = system.energy_per_4_128_128_mxu * origin_num_iters * streaming_dim
        else:
            energy = system.power_per_4_128_128_mxu * self.get_effective_num_ops(system) / (system.op_per_sec)
            power_gated_energy = energy
        return energy, power_gated_energy



    # def get_compute_efficeincy(self, dim_size, mxu_size):
    #     iters = ceil(dim_size/mxu_size)
    #     efficiency = dim_size/ (iters * mxu_size)
    #     return efficiency
    #
    # def get_compute_time(self, system):
    #
    #     if system.mxu_shape is not None:
    #         left, upper, contract, outer = self.get_gemms()
    #         if system.skip_compute:
    #             contract = contract * self.density_w * self.density_a
    #             if contract < 1:
    #                 print(f'[Warning] Contract dimension < 1, after sparsified')
    #             if system.skip_compute_on_noopt_output:
    #                 left = left *self.density_o
    #         mxu_mapping = np.sort([left, upper, contract])[::-1][:2]
    #         effective_mxu_shape = np.sort(system.mxu_shape[-2:])[::-1]
    #         compute_efficiency = np.prod([self.get_compute_efficeincy(d, m) for d, m in zip(mxu_mapping, effective_mxu_shape)])
    #     else:
    #         compute_efficiency = 1.0
    #     return self.get_effective_num_ops(system) * system.get_bit_multiplier(type='C')/system.op_per_sec  / compute_efficiency

    def get_compute_energy(self, system):
        return self.get_effective_num_ops(system)  * system.energy_per_mac

###TODO : Should the sparse pe support be max of individaul density or of the multiple final density.

    def get_effective_num_ops(self, system):
        if system.skip_compute:
            if system.skip_compute_on_noopt_output:
                # return self.get_num_ops() * max(self.density_w * self.density_a * self.density_o , system.sparse_pe_support) 
                return self.get_num_ops() * min(ceil((self.density_w * self.density_a * self.density_o) /system.pe_min_density_support) * system.pe_min_density_support ,1.0)
            else:
                # return self.get_num_ops() * max(self.density_w * self.density_a ,system.sparse_pe_support) 
                return self.get_num_ops() * min( ceil((self.density_w * self.density_a) /system.pe_min_density_support) * system.pe_min_density_support , 1.0)
        else:
            return  self.get_num_ops()

    def get_index_bits_estimator(self, density):
        if density < 0.1:
            bits = 4
        elif density < 0.25:
            bits = 3
        elif density == 1:
            bits = 0
        else:
            bits = 2
        return bits

# The function returns the size of each of the 3 models parameter for each layer, i.e. input, weights and outputs.
    def get_sz_list(self, system=None, index_mem=False):
        if system:
            if system.compress_mem:
                sz_list = [sz * density for sz, density in zip(self.get_sz_list(), self.get_density_list())]
                if not index_mem:                                           # Input ,weight and output are calculated in operators.py for each layer type.
                    return sz_list
                else:                                                       # This is when we are using sparsity and need to add index memory bits to the size.
                    left, upper, contract, outer = self.get_gemms()         # Get loop dimensions
                    contract_w = max(1, contract*self.density_w)
                    contract_a = max(1, contract*self.density_a)
                    index_size_w = upper * contract_w * outer  * self.get_index_bits_estimator(self.density_w) / 8 * system.get_bit_multiplier('M')
                    index_size_a = left * contract_a * outer  * self.get_index_bits_estimator(self.density_a) / 8 * system.get_bit_multiplier('M')
                    sz_list[0] += index_size_a
                    sz_list[1] += index_size_w
                    return sz_list

        return list(map(self.get_size, [self.input_a, self.input_w, self.output]))

    def get_loc_list(self):
        return [self.input_a_loc, self.input_w_loc, self.output_loc]

    def get_memory_time(self, system):
        sz_list = self.get_sz_list(system)
        loc_list = self.get_loc_list()
        memory_time = 0
        # print(sz_list[0],sz_list[1],sz_list[2])i
        # print(self.op_type)
        # print(loc_list[0],loc_list[1],loc_list[2])
        
        if(system.model_on_chip_mem_implications==True and              ## You want to model the on-chip implications, i.e. data refetch
        loc_list==['off','off','off'] and                               ## All your data is off chip
        (min(sz_list[0],sz_list[1])* system.get_bit_multiplier(type='M')) > system.on_chip_mem_size):      ## and smallest input matrix is larger than SRAM
            # for tensor_sz, loc in zip(sz_list, loc_list):
            SRAM = int(system.on_chip_mem_size)/2       # /2 for double buffering
            input_a = int(sz_list[0])
            input_w = int(sz_list[1]) 
            output = int(sz_list[2])
            
            if(self.get_op_type(self.dim) == 'Logit' ):
                num_heads = self.dim[:self.get_effective_dim_len()][1]
                # print(num_heads)
                input_a = input_a/num_heads
                input_w = input_w/num_heads
            elif(self.get_op_type(self.dim) == 'Attend'):
                num_heads = self.dim[:self.get_effective_dim_len()][1]
                input_w = input_w/num_heads
                output = output/num_heads
                # memory_amount
            # else:
            # memory_amount =      ceil(min(input_a,input_w)/SRAM)*(min(input_w,input_a,SRAM)+max(input_a,input_w)) + output
            memory_amount = min(ceil(min(input_a,input_w)/SRAM)*max(input_a,input_w) + min(input_a,input_w) ,ceil(max(input_a,input_w)/SRAM)*min(input_a,input_w)+max(input_a,input_w)) + output
            # print(input_a,input_w,output)
            memory_time = memory_amount* system.get_bit_multiplier(type='M')/system.offchip_mem_bw
            # print(memory_time,memory_amount)
            if(self.get_op_type(self.dim) == 'Logit' or self.get_op_type(self.dim) == 'Attend'):
                num_heads = self.dim[:self.get_effective_dim_len()][1]
                memory_time = memory_time*num_heads 
            
        else:                                                    ## Assume infinite memory
            for tensor_sz, loc in zip(sz_list, loc_list):
                if loc == 'off':
                    bw = system.offchip_mem_bw
                elif loc == 'on':
                    bw = system.onchip_mem_bw
                else:
                    raise ValueError(f'Wrong bw allocation: {loc}.')
                memory_time += tensor_sz * system.get_bit_multiplier(type='M')/bw
        return memory_time




    def get_onchip_occupancy(self):
        sz_list = self.get_sz_list()
        loc_list = self.get_loc_list()
        onchip_mem_occupancy = 0
        for tensor_sz, loc in zip(sz_list, loc_list):
            if loc == 'on':
                onchip_mem_occupancy += tensor_sz

        return onchip_mem_occupancy

    def get_roofline(self, system, unit):
        ideal_compute_time = self.get_ideal_compute_time(system=system)
        ideal_complete_offchip_time, ideal_complete_onchip_time = self.get_ideal_memory_time(system=system)
        # x2 for ops -> MAC has 1 multiplication and 1 Addition hence 2.
        num_ops = self.get_effective_num_ops(system) * 2
        num_data = self.get_effective_num_data(system) * system.get_bit_multiplier(type='M')
        op_intensity = num_ops/num_data

        ##TODO: Why max here, are you assuming that compute and mem ops are completely parallel?
        ideal_exec_time_complete_offchip = max(ideal_compute_time, ideal_complete_offchip_time)
        ideal_exec_time_complete_onchip = max(ideal_compute_time, ideal_complete_onchip_time)

        ideal_thrpt_complete_offchip = num_ops/ideal_exec_time_complete_offchip
        ideal_thrpt_complete_onchip = num_ops/ideal_exec_time_complete_onchip

        compute_time, compute_efficiency = self.get_compute_time(system=system)
        mxu_energy, power_gated_mxu_energy = self.get_mxu_energy(system=system)
        
        compute_time /= system.compute_efficiency
        compute_efficiency /= system.compute_efficiency


        memory_time = self.get_memory_time(system=system) / system.memory_efficiency
        exec_time = max(compute_time, memory_time)
        thrpt = num_ops/exec_time
        com_to_mem_ratio = compute_time/memory_time
        boundedness = 'C' if com_to_mem_ratio > 1 else 'M'

        input_a_size, input_w_size, output_size = self.get_sz_list(system)

        # compute_energy = self.get_compute_energy(system)
        # memory_energy = get_memory_energy(self,system)
        # noc_energy = get_noc_energy(self,system)

        _,_,_,energies = get_matmul_access( self, system )
        total_energy = min(energies)
        # saved_energy_rate = (mxu_energy-power_gated_mxu_energy)/mxu_energy
        ret = {
            'Op Type': self.node.operator,
            'Dimension': self.get_tensors(),
            'Bound': boundedness,
            'C/M ratio': com_to_mem_ratio,
            'Op Intensity': op_intensity,
            f'Latency ({unit.unit_time})': unit.raw_to_unit(exec_time, type='T'),
            f'Cycles': exec_time*system.frequency,
            f'C Effcy': compute_efficiency,
            f'Num ops ({unit.unit_flop})': unit.raw_to_unit(num_ops, type='O'),
            f'Input_a ({unit.unit_mem})': unit.raw_to_unit(input_a_size, type='M'),
            f'Input_w ({unit.unit_mem})': unit.raw_to_unit(input_w_size, type='M'),
            f'Output ({unit.unit_mem})': unit.raw_to_unit(output_size, type='M'),
            f'Total Data ({unit.unit_mem})': unit.raw_to_unit(sum(self.get_sz_list(system)), type='M'),
            f'Throughput ({unit.unit_compute})': unit.raw_to_unit(thrpt, type='C'),
            f'Roofline Throughput offchip ({unit.unit_compute})': unit.raw_to_unit(ideal_thrpt_complete_offchip, type='C'),
            f'Roofline Throughput onchip ({unit.unit_compute})': unit.raw_to_unit(ideal_thrpt_complete_onchip, type='C'),
            f'Compute Cycles': compute_time*system.frequency,
            f'Memory Cycles': memory_time*system.frequency,
            f'Sparsity':(1-self.density_w),
            
            # f'MXU energy (uJ)': mxu_energy *1e6,
            # f'PG-MXU energy (uJ)': power_gated_mxu_energy *1e6,
            # f'Total energy (uJ)': power_gated_mxu_energy*1e6,
            # f'Saved energy (%)': saved_energy_rate * 100,
            # f'Compute energy (mJ)': compute_energy *1e3,
            # f'Mem energy (mJ)': memory_energy *1e3 ,
            # f'NoC energy (mJ)': noc_energy *1e3 ,
            f'Total energy (mJ)': total_energy*1e3,
            # f'Onchip Memory Occupancy ({unit.unit_mem})':  unit.raw_to_unit(self.get_onchip_occupancy(), type='M'),
            # f'Left Onchip Memory ({unit.unit_mem})': unit.raw_to_unit(system.claim_onchip_mem(
            #     self.get_onchip_occupancy()), type='M'),
        }

        return ret










