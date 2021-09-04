from category import *
import pandas as pd
import numpy as np
import argparse
import sys
import os
import copy


class Generate:
    def __init__(self, args):
        self.args = args
        self.tran = None
        self.jvm_type = None
        self.operate_category = None
        self.tran_str = None

    def get_tran_v1(self, instruction_list, instruction_no, constant_pool,
                         instruction_stack, local_var, array_var, stack_map_table, target_idx_list):

        cur_instruction_list = instruction_list[instruction_no]
        instruction = cur_instruction_list[0]
        if_throw_idx = False

        if instruction_no in target_idx_list:
            if_throw_idx = True

        tran = 'null'

        if instruction in self.operate_category['cal_0_goto_branch_category']:
            tran = (self.tran_str[instruction]).format(cur_instruction_list[1])
        elif instruction in self.operate_category['cal_1_goto_branch_category']:
            value = instruction_stack.pop(0)
            if 'lookupswitch' == instruction or 'tableswitch' == instruction:
                tran = (self.tran_str[instruction]).format(value[2])
            else:
                tran = (self.tran_str[instruction]).format(value[2], cur_instruction_list[1])
        elif instruction in self.operate_category['cal_2_goto_branch_category']:
            value2 = instruction_stack.pop(0)
            value1 = instruction_stack.pop(0)
            tran = (self.tran_str[instruction]).format(value1[2], value2[2], cur_instruction_list[1])
        elif instruction in self.operate_category['cal_0_return_category']:
            if instruction == 'return':
                instruction_stack = []
                tran = (self.tran_str[instruction])
            else:
                tran = (self.tran_str[instruction]).format(cur_instruction_list[1])
        elif instruction in self.operate_category['cal_1_return_category']:
            value = instruction_stack.pop(0)
            instruction_stack = []
            tran = (self.tran_str[instruction]).format(value[2])
        elif instruction in self.operate_category['cal_0_push_stack_category']:
            if 'load' in instruction:
                load_type = instruction[0]
                if load_type == 'a':
                    load_type = 'reference'
                elif load_type == 'l':
                    load_type = 'J'
                else:
                    load_type = load_type.upper()

                load_slot = '1'
                if load_type == 'J' or load_type == 'D':
                    load_slot = '2'

                if '_' in instruction:
                    instruction_idx = instruction.split('_')
                    instruction = instruction_idx[0]
                    idx = instruction_idx[1]
                else:
                    idx = cur_instruction_list[1]

                instruction_stack.insert(0, [load_type, load_slot, local_var[idx]['value']])
                tran = (self.tran_str[instruction]).format(local_var[idx]['value'], local_var[idx]['name'])

            elif 'const' in instruction:
                const_type = instruction[0]
                if const_type == 'a':
                    const_type = 'reference'
                elif const_type == 'l':
                    const_type = 'J'
                else:
                    const_type = const_type.upper()

                const_slot = '1'
                if const_type == 'D' or const_type == 'J':
                    const_slot = '2'

                const_value = instruction.split('_')[1]
                if const_value == 'm1':
                    const_value = '-1'

                tran = self.tran_str[instruction]
                instruction_stack.insert(0, [const_type, const_slot, const_value])
            elif 'push' in instruction:
                tran = (self.tran_str[instruction]).format(cur_instruction_list[1])
                instruction_stack.insert(0, ['I', '1', cur_instruction_list[1]])
            elif 'ldc' in instruction:
                ldc_value = constant_pool[cur_instruction_list[1]]['value']
                tran = (self.tran_str[instruction]).format(ldc_value)
                type_ = constant_pool[cur_instruction_list[1]]['type']
                if type_ == 'Double':
                    type_ = 'D'
                    slot = '2'
                elif type_ == 'Long':
                    type_ = 'J'
                    slot = '2'
                elif type_ == 'Integer':
                    type_ = 'I'
                    slot = '1'
                elif type_ == 'Float':
                    type_ = 'F'
                    slot = '1'
                else:
                    type_ = 'reference'
                    slot = '1'
                instruction_stack.insert(0, [type_, slot, ldc_value])
            elif 'get' in instruction:
                var_list = self.get_var_details(constant_pool[cur_instruction_list[1]]['value'])
                instruction_stack.insert(0, var_list)
                tran = (self.tran_str[instruction]).format(var_list[2])
            else:
                print('cal_0_push_stack_category error')
                sys.exit(0)
        elif instruction in self.operate_category['cal_1_push_stack_category']:
            if instruction == 'getfield':
                assert instruction_stack[0][0] == 'reference'
                instruction_stack.pop(0)
                var_list = self.get_var_details(constant_pool[cur_instruction_list[1]]['value'])
                instruction_stack.insert(0, var_list)
                tran = (self.tran_str[instruction]).format(var_list[2])
            elif instruction == 'instanceof':
                objectref = instruction_stack.pop(0)
                assert objectref[0] == 'reference'
                constant_value = constant_pool[cur_instruction_list[1]]['value']
                tran = (self.tran_str[instruction]).format(objectref[2], constant_value)
                instruction_stack.insert(0, ['I', '1', 'int constant'])
            elif instruction == 'checkcast':
                tran = (self.tran_str[instruction]).format(instruction_stack[0][2])
            else:
                tran = (self.tran_str[instruction]).format(instruction_stack[0][2])

                if instruction == 'arraylength':
                    instruction_stack.pop(0)
                    instruction_stack.insert(0, ['I', '1', 'arraylength'])
                elif '2' in instruction:

                    new_type = 'J' if instruction[2] == 'l' else instruction[2].upper()
                    instruction_stack[0][0] = new_type
                    instruction_stack[0][1] = self.jvm_type[new_type]['slot']
                elif 'neg' in instruction:
                    instruction_stack[0][2] = '- ' + instruction_stack[0][2]
        elif instruction in self.operate_category['cal_2_push_stack_category']:
            if 'load' in instruction:
                array_value = 'value'
                index = instruction_stack.pop(0)
                arrayref = instruction_stack.pop(0)
                if arrayref[2] in array_var.keys():
                    if index[2] in array_var[arrayref[2]]:
                        array_value = array_var[arrayref[2]][index[2]]
                tran = (self.tran_str[instruction]).format(array_value, arrayref[2], index[2])
                type_ = 'J' if instruction[0] == 'l' else instruction[0].upper()
                type_ = 'reference' if type_ == 'A' else type_
                slot = '2' if type_ == 'J' or type_ == 'D' else '1'
                instruction_stack.insert(0, [type_, slot, array_value])
            else:
                tran = (self.tran_str[instruction]).format(instruction_stack[0][2], instruction_stack[1][2])
                if 'swap' == instruction:
                    value1 = instruction_stack.pop(0)
                    value2 = instruction_stack.pop(0)
                    instruction_stack.insert(0, value1)
                    instruction_stack.insert(0, value2)
                elif 'cmp' in instruction:
                    instruction_stack.pop(0)
                    instruction_stack.pop(0)
                    instruction_stack.insert(0, ['I', '1', 'value'])
                else:
                    instruction_stack.pop(0)
                    instruction_stack.pop(0)
                    type_ = 'J' if instruction[0] == 'l' else instruction[0].upper()
                    slot = '2' if type_ == 'J' or type_ == 'D' else '1'
                    instruction_stack.insert(0, [type_, slot, 'value'])
        elif instruction in self.operate_category['cal_3_push_stack_category']:
            pass
        elif instruction in self.operate_category['cal_n_push_stack_category']:
            method_origin = constant_pool[cur_instruction_list[1]]['value']
            method_name, method_args, method_return = self.get_method_details(method_origin)
            for type_ in method_args[::-1]:
                instruction_stack.pop(0)

            if instruction == 'invokeinterface' or instruction == 'invokespecial' or instruction == 'invokevirtual':
                assert instruction_stack[0][0] == 'reference'
                instruction_stack.pop(0)

            if method_return != 'V':
                if method_return in self.jvm_type.keys():
                    instruction_stack.insert(0, [method_return, self.jvm_type[method_return]['slot'], method_return])
                else:
                    instruction_stack.insert(0, ['reference', '1', method_return])
            tran = (self.tran_str[instruction]).format(method_name)
        elif instruction in self.operate_category['cal_0_store_var_category']:
            var_ = local_var[cur_instruction_list[1]]['value']
            constant_ = cur_instruction_list[2]
            tran = (self.tran_str[instruction]).format(var_, constant_)
            if var_.isdigit():
                new_var_ = str(int(var_) + int(constant_))
                local_var[cur_instruction_list[1]]['value'] = new_var_
        elif instruction in self.operate_category['cal_1_store_var_category']:
            if 'store' in instruction:
                if '_' in instruction:
                    instruction_idx = instruction.split('_')
                    instruction = instruction_idx[0]
                    idx = instruction_idx[1]
                else:
                    idx = cur_instruction_list[1]

                if if_throw_idx:
                    instruction_stack = []
                    if idx not in local_var.keys():
                        local_var.update({idx: {'start': '', 'length': '', 'name': f'var{idx}',
                                                'signature': 'thrown signature', 'value': 'thrown value'}})
                    tran = (self.tran_str[instruction]).format('thrown value', local_var[idx]['name'])
                else:
                    if idx not in local_var.keys():
                        local_var.update({idx: {'start': '', 'length': '', 'name': f'var{idx}',
                                                'signature': instruction_stack[0][2], 'value': instruction_stack[0][2]}})
                    tran = (self.tran_str[instruction]).format(instruction_stack[0][2], local_var[idx]['name'])
                    local_var[idx][2] = instruction_stack.pop(0)
            elif 'put' in instruction:
                value = instruction_stack.pop(0)
                tran = (self.tran_str[instruction]).format(value[2], constant_pool[cur_instruction_list[1]]['value'])
            else:
                print('cal_1_store_var_category error')
                sys.exit(0)
        elif instruction in self.operate_category['cal_2_store_var_category']:
            value = instruction_stack.pop(0)[2]
            objectref = instruction_stack.pop(0)[2]
            constant_pool = self.set_var_value(constant_pool, cur_instruction_list[1], value)
            tran = (self.tran_str[instruction]).format(value, objectref)
        elif instruction in self.operate_category['cal_3_store_var_category']:
            array_value = instruction_stack.pop(0)[2]
            array_idx = instruction_stack.pop(0)[2]
            array_ = instruction_stack.pop(0)[2]
            tran = (self.tran_str[instruction]).format(array_value, array_, array_idx)
            if array_ not in array_var.keys():
                array_var[array_] = {}
            array_var[array_][array_idx] = array_value
        elif instruction in self.operate_category['cal_0_create_category']:
            value = constant_pool[cur_instruction_list[1]]['value']
            instruction_stack.insert(0, ['reference', '1', value])
            tran = (self.tran_str[instruction]).format(value)
        elif instruction in self.operate_category['cal_1_create_category']:
            instruction_stack.pop(0)
            array_value = cur_instruction_list[1]
            if instruction.startswith('a'):
                array_value = constant_pool[array_value]['value']
                instruction_stack.insert(0, ['reference', '1', array_value])
            else:
                instruction_stack.insert(0, ['reference', '1', array_value + ' array'])
            tran = (self.tran_str[instruction]).format(array_value)
        elif instruction in self.operate_category['cal_n_create_category']:
            for i in range(int(cur_instruction_list[2])):
                instruction_stack.pop(0)
            array_type = constant_pool[cur_instruction_list[1]]['value'].replace('[', '')
            if array_type in self.jvm_type.keys():
                array_type = self.jvm_type[array_type]['act_type']
            else:
                array_type = 'reference'
            instruction_stack.insert(0, ['reference', '1', 'multianewarray'])
            tran = (self.tran_str[instruction]).format(array_type)
        elif instruction in self.operate_category['management_stack_category']:
            if 'pop' in instruction:
                if 'pop2' == instruction:
                    if instruction_stack[0][1] == '2':
                        tran = (self.tran_str[f'{instruction}_0']).format(instruction_stack[0][2])
                        instruction_stack.pop(0)
                    elif instruction_stack[0][1] == '1':
                        if instruction_stack[1][1] == '1':
                            tran = (self.tran_str[f'{instruction}_0']).format(instruction_stack[0][2],
                                                                                   instruction_stack[1][2])
                            instruction_stack.pop(0)
                            instruction_stack.pop(0)
                        else:
                            print('pop2_1 error')
                            sys.exit(0)
                    else:
                        print('pop2 error')
                        sys.exit(0)
                else:
                    tran = (self.tran_str[instruction]).format(instruction_stack[0][2])
                    instruction_stack.pop(0)
            elif 'dup' in instruction:
                if 'dup2' in instruction:
                    x_ = instruction.split('x')
                    site_ = 2 if len(x_) == 1 else int(x_[1]) + 2
                    if instruction_stack[0][1] == '2':
                        slot_sum = 0
                        for idx, i in enumerate(instruction_stack):
                            slot_sum += int(i[1])
                            if slot_sum == site_:
                                instruction_stack.insert(idx + 1, instruction_stack[0])
                                break
                            elif slot_sum > site_:
                                print('dup2 error')
                                sys.exit(0)
                        tran = (self.tran_str[instruction + '_1']).format(instruction_stack[0][2])
                    elif instruction_stack[0][1] == '1' and instruction_stack[1][1] == '1':
                        slot_sum = 0
                        for idx, i in enumerate(instruction_stack):
                            slot_sum += int(i[1])
                            if slot_sum == site_:
                                instruction_stack.insert(idx + 1, instruction_stack[1])
                                instruction_stack.insert(idx + 1, instruction_stack[0])
                                break
                            elif slot_sum > site_:
                                print('dup2 error')
                                sys.exit(0)
                        tran = (self.tran_str[instruction + '_2']).format(instruction_stack[0][2], instruction_stack[1][2])
                else:
                    x_ = instruction.split('x')
                    site_ = 1 if len(x_) == 1 else int(x_[1]) + 1
                    slot_sum = 0
                    for idx, i in enumerate(instruction_stack):
                        slot_sum += int(i[1])
                        if slot_sum == site_:
                            instruction_stack.insert(idx + 1, instruction_stack[0])
                            break
                        elif slot_sum > site_:
                            print('dup error')
                            sys.exit(0)
                    tran = (self.tran_str[instruction]).format(instruction_stack[0][2])
        elif instruction in self.operate_category['cal_1_monitor_category']:
            tran = (self.tran_str[instruction]).format(instruction_stack[0][2])
            instruction_stack.pop(0)
        elif instruction in self.operate_category['do_nothing_category']:
            tran = (self.tran_str[instruction])
        elif instruction in self.operate_category['unknow_category']:
            print('unknow category error')
            sys.exit(0)
        else:
            print('instruction category error')
            print(instruction)
            sys.exit(0)

        return tran, constant_pool, instruction_stack, local_var, array_var

    def get_tran_v2(self, instruction):
        print('tran v2 to be continued')
        sys.exit(0)

    def get_instruction_branch(self, instruction_origin_dict, instruction_idx_list, cur_idx):
        branch1 = []
        branch2 = []
        cur_instruction_idx = instruction_idx_list[cur_idx]

        if cur_idx == len(instruction_idx_list) - 1 or 'return' in instruction_origin_dict[cur_instruction_idx][0] or 'goto' in \
                instruction_origin_dict[cur_instruction_idx][0]:
            if 'goto' in instruction_origin_dict[cur_instruction_idx][0]:
                if int(instruction_origin_dict[cur_instruction_idx][1]) > int(cur_instruction_idx):
                    goto_idx = instruction_origin_dict[cur_instruction_idx][1]
                    branch1 = [cur_instruction_idx + ' ' + i for i in
                               self.get_instruction_branch(instruction_origin_dict, instruction_idx_list,
                                                      instruction_idx_list.index(goto_idx))]
                else:
                    return [cur_instruction_idx]
            else:
                return [cur_instruction_idx]
        else:
            branch1 = [cur_instruction_idx + ' ' + i for i in
                       self.get_instruction_branch(instruction_origin_dict, instruction_idx_list, cur_idx + 1)]

        if 'if' in instruction_origin_dict[cur_instruction_idx][0] or 'goto' in instruction_origin_dict[cur_instruction_idx][0]:
            if int(instruction_origin_dict[cur_instruction_idx][1]) < int(cur_instruction_idx):
                return [cur_instruction_idx]
            else:
                goto_idx = instruction_origin_dict[cur_instruction_idx][1]
                branch2 = [cur_instruction_idx + ' ' + i for i in
                           self.get_instruction_branch(instruction_origin_dict, instruction_idx_list, instruction_idx_list.index(goto_idx))]

        return branch1 + branch2

    def set_var_value(self, constant_pool, idx, value):
        pre_value = constant_pool[idx]['value']
        value_name = pre_value.split(':')[0]
        new_value = value_name + ':' + value
        constant_pool[idx]['value'] = new_value
        return constant_pool

    def get_var_details(self, var_str):
        var_tn_str = var_str.split('.')[-1]
        var_tn_list = var_tn_str.split(':')
        var_name = var_tn_list[0]
        var_type = var_tn_list[1]
        var_slot = '1'

        if var_type not in self.jvm_type.keys() and (var_type != 'J' or var_type != 'D'):
            var_type = 'reference'
        elif var_type == 'J' or var_type == 'D':
            var_slot = '2'

        return [var_type, var_slot, var_name]

    def get_method_details(self, method_str):
        method_str_list = method_str.split(':(')
        assert len(method_str_list) == 2

        method_name = method_str_list[0].split('.')[-1]
        method_args_return = method_str_list[1].split(')')
        assert len(method_args_return) == 2

        method_args = method_args_return[0]
        method_return = method_args_return[1]
        method_args_list = []
        args_str = ''
        is_arg = False
        for c in method_args:
            if c == ';':
                args_str += c
                method_args_list.append(args_str)
                args_str = ''
                is_arg = False
            elif c in self.jvm_type.keys():
                args_str += c
                if not is_arg:
                    method_args_list.append(args_str)
                    args_str = ''
            else:
                if c == 'L' or '[L' in args_str:
                    is_arg = True
                args_str += c

        return method_name, method_args_list, method_return

    def get_localVariableTable(self, lines):
        localVariableTable = {}
        for line in lines[2:]:
            line_list = line.split()
            if not line_list[0].isdigit():
                break
            localVariableTable[line_list[2]] = {}
            localVariableTable[line_list[2]]['start'] = line_list[0]
            localVariableTable[line_list[2]]['length'] = line_list[1]
            localVariableTable[line_list[2]]['name'] = line_list[3]
            localVariableTable[line_list[2]]['signature'] = line_list[4]
            localVariableTable[line_list[2]]['value'] = line_list[3]
        return localVariableTable

    def get_constantPool(self, lines):
        constantPool = {}
        for line in lines:
            line_list = line.split()
            constantPool[line_list[0]] = {}
            constantPool[line_list[0]]['type'] = line_list[2]
            if line_list[2] == 'Utf8':
                constantPool[line_list[0]]['value'] = ' '.join(line_list[3:])
            elif line_list[2] == 'String':
                constantPool[line_list[0]]['value'] = 'string'
            else:
                index = 0
                for idx, i in enumerate(line_list[3:]):
                    if i.startswith('//'):
                        index = idx + 1 + 3
                        break
                constantPool[line_list[0]]['value'] = ' '.join(line_list[index:])

        return constantPool

    def get_exceptionTable(self, lines):
        exceptionTable = []
        target_idx_list = []
        for line in lines[2:]:
            line_list = line.split()
            exceptionTable.append([line_list[0], line_list[1], line_list[2], ' '.join(line_list[3:])])
            target_idx_list.append(line_list[2])
        return exceptionTable, target_idx_list

    def get_lineNumberTable(self, lines):
        pass

    def get_stackMapTable(self, lines):
        pass

    def instruction_to_tran(self, method_name, lines):
        constantPool_s_idx = 0
        constantPool_e_idx = 0
        for idx, line in enumerate(lines):
            if line.startswith('Constant pool:'):
                constantPool_s_idx = idx

            if line.startswith('{'):
                constantPool_e_idx = idx - 1
                break
        constant_pool = self.get_constantPool(lines[constantPool_s_idx + 1:constantPool_e_idx + 1])

        method_instruction_s_idx = 0
        method_instruction_e_idx = len(lines) - 1
        for idx, line in enumerate(lines[constantPool_e_idx + 1:]):
            if line.startswith('    descriptor: ('):
                method_instruction_s_idx = idx + constantPool_e_idx

        lines = lines[method_instruction_s_idx:method_instruction_e_idx + 1]
        if ' ' + method_name + '(' not in lines[0]:
            return np.nan

        exceptionTable_s_idx = 0
        exceptionTable_e_idx = 0
        for idx, line in enumerate(lines):
            if line.startswith('      Exception table:'):
                exceptionTable_s_idx = idx
            if line.startswith('      LineNumberTable:') or line.startswith('      LocalVariableTable:'):
                if exceptionTable_s_idx == 0:
                    break
                exceptionTable_e_idx = idx
                break
        exception_table, target_idx_list = self.get_exceptionTable(lines[exceptionTable_s_idx:exceptionTable_e_idx])

        lineNumberTable_s_idx = exceptionTable_e_idx
        lineNumberTable_e_idx = exceptionTable_e_idx
        for idx, line in enumerate(lines[exceptionTable_e_idx:]):
            if line.startswith('      LineNumberTable:'):
                lineNumberTable_s_idx += idx
            if line.startswith('      LocalVariableTable:'):
                lineNumberTable_e_idx += idx
                break
        line_number_table = self.get_lineNumberTable(lines[lineNumberTable_s_idx:lineNumberTable_e_idx])

        localVariableTable_s_idx = lineNumberTable_e_idx
        localVariableTable_e_idx = lineNumberTable_e_idx
        for idx, line in enumerate(lines[lineNumberTable_e_idx:]):
            if line.startswith('      LocalVariableTable:'):
                localVariableTable_s_idx += idx
            if line.startswith('    Exceptions:') or line.startswith('      StackMapTable:') or line.startswith(
                    '}'):
                localVariableTable_e_idx += idx
                break
        local_var = self.get_localVariableTable(lines[localVariableTable_s_idx:localVariableTable_e_idx])
        stack_map_table = self.get_stackMapTable(lines[localVariableTable_e_idx:])

        instruction_origin_dict = {}

        if exceptionTable_s_idx == 0:
            if lineNumberTable_s_idx == 0:
                instruction_e_idx = localVariableTable_s_idx
            else:
                instruction_e_idx = lineNumberTable_s_idx
        else:
            instruction_e_idx = exceptionTable_s_idx

        is_switch_instruction = False
        for line in lines[5:instruction_e_idx]:
            instruction_list = line.split()
            if is_switch_instruction:
                if len(instruction_list) == 1 and instruction_list[0] == '}':
                    is_switch_instruction = False
                continue

            instruction_no = instruction_list[0][:-1]
            instruction = instruction_list[1]
            if instruction == 'lookupswitch' or instruction == 'tableswitch':
                is_switch_instruction = True

            instruction_origin_dict[instruction_no] = []
            instruction_origin_dict[instruction_no].append(instruction)

            if len(instruction_list) > 1:
                instruction_remainder = instruction_list[2:]
                for idx, i in enumerate(instruction_remainder):
                    if i.startswith(r'//'):
                        instruction_annotation = instruction_remainder[idx + 1:]
                        instruction_origin_dict[instruction_no].append(' '.join(instruction_annotation))
                        break
                    instruction_origin_dict[instruction_no].append(i.replace(',', ''))
            else:
                print('instruction_list error')
                sys.exit(0)

        if_num = 0
        for ood in instruction_origin_dict.values():
            if 'if' in ood[0]:
                if_num += 1

        if if_num > 20:
            return np.nan

        instruction_idx_list = list(instruction_origin_dict.keys())
        try:
            instruction_branch_list = self.get_instruction_branch(instruction_origin_dict, instruction_idx_list, 0)
        except Exception as e:
            return np.nan

        main_branch_list = instruction_branch_list[0].split()
        main_branch_set = set(main_branch_list)

        new_instruction_branch_list = [main_branch_list]

        for instruction_branch_str in instruction_branch_list[1:]:
            branch_list = instruction_branch_str.split()
            branch_set = set(branch_list)
            if not main_branch_set >= branch_set:
                new_instruction_branch_list.append(branch_list)
                main_branch_set = main_branch_set.union(branch_set)

        array_var = {}
        tran_dict = {}

        for bl in new_instruction_branch_list:
            instruction_stack = []
            visited_instruction_no = []
            visited_instruction_stack = {}
            for idx, instruction_no in enumerate(bl):
                if instruction_no not in visited_instruction_no:
                    if idx != 0:
                        instruction_stack = copy.deepcopy(visited_instruction_stack[bl[idx - 1]])
                    tran, constant_pool, instruction_stack, local_var, array_var = self.get_tran_v1(
                        instruction_origin_dict, instruction_no,
                        constant_pool, instruction_stack,
                        local_var, array_var,
                        stack_map_table,
                        target_idx_list)
                    visited_instruction_no.append(instruction_no)
                    visited_instruction_stack[instruction_no] = instruction_stack
                    tran_dict[int(instruction_no)] = tran
        order_tran = sorted(tran_dict.items(), key=lambda i: i[0], reverse=False)
        tran_list = [v for k, v in order_tran]
        return tran_list

    def preprocess(self):

        df_root_dir = args.root_dir + args.input_path
        if args.process_all:
            df_files = os.listdir(df_root_dir)
            df_files = [i for i in df_files if i.endswith('.pkl')]
            for file_name in df_files:
                pd.read_pickle(df_root_dir + file_name)
        else:
            df_file = df_root_dir + args.process_file
            df = pd.read_pickle(df_file)

            ii = 0

            tran_list = []
            for name, instruction in zip(df['name'], df['instruction']):

                name = 'getSum'
                instruction_file = open('javap_GetSumFor.txt')

                method_name = name.split('_')[0]

                lines = instruction_file.readlines()

                tran = self.instruction_to_tran(method_name, lines)
                tran_list.append(tran)

                print(f'{ii} {method_name}')

                ii += 1

            df['tran'] = tran_list
            new_df = df.dropna(axis=0, how='any')
            new_df.to_pickle(args.root_dir + args.output_path + args.process_file)
            print('old df num', len(df))
            print('new df num', len(new_df))

    def set_tran_category(self):
        self.jvm_type = JVM_type()
        self.operate_category = operate_Category()
        self.tran_str = tran_str()

    def run(self):
        self.set_tran_category()
        self.preprocess()


def parse_args():
    parser = argparse.ArgumentParser("Prepare dataset for TranCS")
    parser.add_argument('--root_dir', type=str, default='../get_from_dataset/dataset/')
    parser.add_argument('--input_path', type=str, default='df/')
    parser.add_argument('--output_path', type=str, default='op/')

    parser.add_argument('--docs_file', type=str, default='../get_from_specification/preprocessed_docs/docs_8.json')

    parser.add_argument('--all_file', type=str, default='all_instruction.txt')
    parser.add_argument('--train_file', type=str, default='train.txt')
    parser.add_argument('--process_all', type=bool, default=False)
    parser.add_argument('--process_file', type=str, default='xxx.pkl')

    parser.add_argument('--dirs_num', type=int, default=1)

    parser.add_argument('--shuffle_index_file', type=str, default='shuffle_index.npy')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate = Generate(args)
    generate.run()
