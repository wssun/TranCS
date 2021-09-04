def JVM_type():
    type = {
        'Z': {'act_type': 'boolean', 'com_type': 'int', 'cate': '1', 'slot': '1'},
        'B': {'act_type': 'byte', 'com_type': 'int', 'cate': '1', 'slot': '1'},
        'C': {'act_type': 'char', 'com_type': 'int', 'cate': '1', 'slot': '1'},
        'S': {'act_type': 'short', 'com_type': 'int', 'cate': '1', 'slot': '1'},
        'I': {'act_type': 'int', 'com_type': 'int', 'cate': '1', 'slot': '1'},
        'F': {'act_type': 'float', 'com_type': 'int', 'cate': '1', 'slot': '1'},
        'reference': {'act_type': 'reference', 'com_type': 'reference', 'cate': '1', 'slot': '1'},
        'returnAddress': {'act_type': 'returnAddress', 'com_type': 'returnAddress', 'cate': '1', 'slot': '1'},
        'J': {'act_type': 'long', 'com_type': 'double', 'cate': '2', 'slot': '2'},
        'D': {'act_type': 'double', 'com_type': 'double', 'cate': '2', 'slot': '2'},
        'V': {'slot': 0},
    }
    return type


def instructions():
    instructions = [
        'aaload', 'baload', 'caload', 'daload', 'faload', 'iaload', 'laload', 'saload',
        'aastore', 'bastore', 'castore', 'dastore', 'fastore', 'iastore', 'lastore', 'sastore',

        'aload', 'aload_0', 'aload_1', 'aload_2', 'aload_3',
        'dload', 'dload_0', 'dload_1', 'dload_2', 'dload_3',
        'fload', 'fload_0', 'fload_1', 'fload_2', 'fload_3',
        'iload', 'iload_0', 'iload_1', 'iload_2', 'iload_3',
        'lload', 'lload_0', 'lload_1', 'lload_2', 'lload_3',
        'astore', 'astore_0', 'astore_1', 'astore_2', 'astore_3',
        'dstore', 'dstore_0', 'dstore_1', 'dstore_2', 'dstore_3',
        'fstore', 'fstore_0', 'fstore_1', 'fstore_2', 'fstore_3',
        'istore', 'istore_0', 'istore_1', 'istore_2', 'istore_3',
        'lstore', 'lstore_0', 'lstore_1', 'lstore_2', 'lstore_3',

        'bipush', 'sipush',
        'putfield', 'putstatic',

        'aconst_null',
        'dconst_0', 'dconst_1',
        'fconst_0', 'fconst_1', 'fconst_2',
        'iconst_m1', 'iconst_0', 'iconst_1', 'iconst_2', 'iconst_3', 'iconst_4, iconst_5',
        'lconst_0', 'lconst_1',

        'invokedynamic', 'invokeinterface', 'invokespecial', 'invokestatic', 'invokevirtual',

        'd2f', 'd2i', 'd2l',
        'f2d', 'f2i', 'f2l',
        'i2b', 'i2c', 'i2d', 'i2f', 'i2l', 'i2s',
        'l2d', 'l2f', 'l2i',

        'dadd', 'ddiv', 'dmul', 'dneg', 'drem', 'dsub',
        'fadd', 'fdiv', 'fmul', 'fneg', 'frem', 'fsub',
        'iadd', 'idiv', 'imul', 'ineg', 'irem', 'isub',
        'ladd', 'ldiv', 'lmul', 'lneg', 'lrem', 'lsub',

        'iand', 'ior', 'ixor',
        'ishl', 'ishr', 'iushr',
        'land', 'lor', 'lxor',
        'lshl', 'lshr', 'lushr',

        'ldc', 'ldc_w', 'ldc2_w',

        'areturn', 'dreturn', 'freturn', 'ireturn', 'lreturn', 'ret', 'return',

        'dcmpg', 'dcmpl', 'fcmpg', 'fcmpl', 'lcmp',

        'if_acmpeq', 'if_acmpne',
        'if_icmpeq', 'if_icmpne', 'if_icmplt', 'if_icmpge', 'if_icmpgt', 'if_icmple',
        'ifeq', 'ifne', 'iflt', 'ifge', 'ifgt', 'ifle',
        'ifnonnull', 'ifnull',

        'goto', 'goto_w',
        'jsr', 'jsr_w',

        'lookupswitch',
        'tableswitch',

        'dup', 'dup_x1', 'dup_x2', 'dup2', 'dup2_x1', 'dup2_x2',

        'monitorenter', 'monitorexit',

        'iinc',
        'athrow',
        'anewarray', 'arraylength', 'multianewarray', 'newarray',
        'new',
        'nop',
        'pop', 'pop2',
        'swap',
        'wide',
        'checkcast',
        'getfield', 'getstatic',
        'instanceof',

    ]
    return instructions


def tran_str():
    instruction2tran = {
        'goto': 'go to {}',
        'goto_w': 'go to {}',
        'jsr': 'jump subroutine {}',
        'jsr_w': 'jump subroutine {}',

        'ifeq': 'if and only if {} is equal to 0 then go to {}',
        'ifne': 'if and only if {} is not equal to 0 then go to {}',
        'iflt': 'if and only if {} is less then 0 then go to {}',
        'ifle': 'if and only if {} is less or equal to 0 then go to {}',
        'ifgt': 'if and only if {} is greater than 0 then go to {}',
        'ifge': 'if and only if {} is greater or equal to 0 then go to {}',
        'ifnull': 'if {} is null then go to {}',
        'ifnonnull': 'if {} is not null then go to {}',

        'lookupswitch': 'access jump table by {}',
        'tableswitch': 'access jump table by {}',

        'if_acmpeq': 'if and only if {} is equal to {} then go to {}',
        'if_acmpne': 'if and only if {} is not equal to {} then go to {}',
        'if_icmpeq': 'if and only if {} is equal to {} then go to {}',
        'if_icmpne': 'if and only if {} is not equal to {} then go to {}',
        'if_icmplt': 'if and only if {} is less then {} then go to {}',
        'if_icmple': 'if and only if {} is less or equal to {} then go to {}',
        'if_icmpgt': 'if and only if {} is greater than {} then go to {}',
        'if_icmpge': 'if and only if {} is greater or equal to {} then go to {}',

        'ret': 'return from subroutine {}',
        'return': 'return void from method',

        'areturn': 'return reference {} from method',
        'dreturn': 'return double {} from method',
        'freturn': 'return float {} from method',
        'ireturn': 'return int {} from method',
        'lreturn': 'return long {} from method',

        'aload': 'load reference {} from local variable {}',
        'aload_0': 'load reference {} from local variable {}',
        'aload_1': 'load reference {} from local variable {}',
        'aload_2': 'load reference {} from local variable {}',
        'aload_3': 'load reference {} from local variable {}',

        'dload': 'load double {} from local variable {}',
        'dload_0': 'load double {} from local variable {}',
        'dload_1': 'load double {} from local variable {}',
        'dload_2': 'load double {} from local variable {}',
        'dload_3': 'load double {} from local variable {}',

        'fload': 'load float {} from local variable {}',
        'fload_0': 'load float {} from local variable {}',
        'fload_1': 'load float {} from local variable {}',
        'fload_2': 'load float {} from local variable {}',
        'fload_3': 'load float {} from local variable {}',

        'iload': 'load int {} from local variable {}',
        'iload_0': 'load int {} from local variable {}',
        'iload_1': 'load int {} from local variable {}',
        'iload_2': 'load int {} from local variable {}',
        'iload_3': 'load int {} from local variable {}',

        'lload': 'load long {} from local variable {}',
        'lload_0': 'load long {} from local variable {}',
        'lload_1': 'load long {} from local variable {}',
        'lload_2': 'load long {} from local variable {}',
        'lload_3': 'load long {} from local variable {}',

        'aconst_null': 'push null',
        'dconst_0': 'push double constant 0.0',
        'dconst_1': 'push double constant 1.0',
        'fconst_0': 'push float constant 0.0',
        'fconst_1': 'push float constant 1.0',
        'fconst_2': 'push float constant 2.0',
        'iconst_m1': 'push int constant -1',
        'iconst_0': 'push int constant 0',
        'iconst_1': 'push int constant 1',
        'iconst_2': 'push int constant 2',
        'iconst_3': 'push int constant 3',
        'iconst_4': 'push int constant 4',
        'iconst_5': 'push int constant 5',
        'lconst_0': 'push long constant 0',
        'lconst_1': 'push long constant 1',

        'bipush': 'push int {}',
        'sipush': 'push int {}',

        'ldc': 'push {} from run-time constant pool',
        'ldc_w': 'push {} from run-time constant pool',
        'ldc2_w': 'push long or double {} from run-time constant pool',

        'getstatic': 'get static field from class {}',

        'd2f': 'convert double {} to float',
        'd2i': 'convert double {} to int',
        'd2l': 'convert double {} to long',
        'f2d': 'convert float {} to double',
        'f2i': 'convert float {} to int',
        'f2l': 'convert float {} to long',
        'i2b': 'convert int {} to byte',
        'i2c': 'convert int {} to char',
        'i2d': 'convert int {} to double',
        'i2f': 'convert int {} to float',
        'i2l': 'convert int {} to long',
        'i2s': 'convert int {} to short',
        'l2d': 'convert long {} to double',
        'l2f': 'convert long {} to float',
        'l2i': 'convert long {} to int',

        'dneg': 'negate double {}',
        'fneg': 'negate float {}',
        'ineg': 'negate int {}',
        'lneg': 'negate long {}',

        'instanceof': 'object {} is of given type {}',
        'checkcast': 'check whether {} is of given type',
        'arraylength': 'get length of array {}',
        'athrow': 'throw exception or error from {}',

        'getfield': 'fetch field from object {}',

        'aaload': 'load reference {} from array {}[{}]',
        'baload': 'load int {} from array {}[{}]',
        'caload': 'load char {} from array {}[{}]',
        'daload': 'load double {} from array {}[{}]',
        'faload': 'load float {} from array {}[{}]',
        'iaload': 'load int {} from array {}[{}]',
        'laload': 'load long {} from array {}[{}]',
        'saload': 'load short {} from array {}[{}]',
        'dadd': 'double {} add double {}',
        'ddiv': 'double {} divided by double {}',
        'dmul': 'double {} multiply double {}',
        'drem': 'double {} remainder double {}',
        'dsub': 'double {} substract double {}',
        'fadd': 'float {} add float {}',
        'fdiv': 'float {} divided by float {}',
        'fmul': 'float {} multiply float {}',
        'frem': 'float {} remainder float {}',
        'fsub': 'float {} substract float {}',
        'iadd': 'int {} add int {}',
        'idiv': 'int {} divided by int {}',
        'imul': 'int {} multiply int {}',
        'irem': 'int {} remainder int {}',
        'isub': 'int {} substract int {}',
        'ladd': 'long {} add long {}',
        'ldiv': 'long {} divided by long {}',
        'lmul': 'long {} multiply long {}',
        'lrem': 'long {} remainder long {}',
        'lsub': 'long {} substract long {}',

        'iand': 'take bitwise AND of int {} and {}',
        'ior': 'take bitwise inclusive OR of int {} and {}',
        'ixor': 'take bitwise exclusive OR of int {} and {}',
        'ishl': 'int {} shift {} left',
        'ishr': 'int {} arithmetic shift {} right',
        'iushr': 'int {} logical shift {} right',
        'land': 'take bitwise AND of long {} and {}',
        'lor': 'take bitwise inclusive OR of long {} and {}',
        'lxor': 'take bitwise exclusive OR of long {} and {}',
        'lshl': 'long {} shift {} left',
        'lshr': 'long {} arithmetic shift {} right',
        'lushr': 'long {} logical shift {} right',
        'dcmpg': 'compare double {} and {}',
        'dcmpl': 'compare double {} and {}',
        'fcmpg': 'compare float {} and {}',
        'fcmpl': 'compare float {} and {}',
        'lcmp': 'compare long {} and {}',

        'swap': 'swap {} and {}',

        'invokedynamic': 'invoke dynamic method {}',
        'invokestatic': 'invoke class static method {}',
        'invokeinterface': 'invoke interface method {}',
        'invokespecial': 'invoke instance superclass, private or instance initialization method {}',
        'invokevirtual': 'invoke instance method {}',

        'iinc': 'increment local variable {} by constant {}',
        'iinc_w': 'increment local variable {} by constant {}',

        'astore': 'store reference {} into local variable {}',
        'astore_0': 'store reference {} into local variable {}',
        'astore_1': 'store reference {} into local variable {}',
        'astore_2': 'store reference {} into local variable {}',
        'astore_3': 'store reference {} into local variable {}',
        'dstore': 'store double {} into local variable {}',
        'dstore_0': 'store double {} into local variable {}',
        'dstore_1': 'store double {} into local variable {}',
        'dstore_2': 'store double {} into local variable {}',
        'dstore_3': 'store double {} into local variable {}',
        'fstore': 'store float {} into local variable {}',
        'fstore_0': 'store float {} into local variable {}',
        'fstore_1': 'store float {} into local variable {}',
        'fstore_2': 'store float {} into local variable {}',
        'fstore_3': 'store float {} into local variable {}',
        'istore': 'store int {} into local variable {}',
        'istore_0': 'store int {} into local variable {}',
        'istore_1': 'store int {} into local variable {}',
        'istore_2': 'store int {} into local variable {}',
        'istore_3': 'store int {} into local variable {}',
        'lstore': 'store long {} into local variable {}',
        'lstore_0': 'store long {} into local variable {}',
        'lstore_1': 'store long {} into local variable {}',
        'lstore_2': 'store long {} into local variable {}',
        'lstore_3': 'store long {} into local variable {}',

        'putstatic': 'set static {} in class {}',

        'aastore': 'store {} into reference array {}[{}]',
        'bastore': 'store {} into int array {}[{}]',
        'castore': 'store {} into char array {}[{}]',
        'dastore': 'store {} into double array {}[{}]',
        'fastore': 'store {} into float array {}[{}]',
        'iastore': 'store {} into int array {}[{}]',
        'lastore': 'store {} into long array {}[{}]',
        'sastore': 'store {} into short array {}[{}]',

        'putfield': 'set field {} in object {}',

        'new': 'create new object {}',

        'anewarray': 'create new array of reference {}',
        'newarray': 'create new {} array',

        'multianewarray': 'create new {} multidimensional array',


        'pop': 'pop {}',

        'pop2_0': 'pop {}',
        'pop2_1': 'pop {} and {}',
        'dup': 'duplicate {}',
        'dup_x1': 'duplicate {}',
        'dup_x2': 'duplicate {}',

        'dup2_1': 'duplicate {}',
        'dup2_x1_1': 'duplicate {}',
        'dup2_x2_1': 'duplicate {}',
        'dup2_2': 'duplicate {} and {}',
        'dup2_x1_2': 'duplicate {} and {}',
        'dup2_x2_2': 'duplicate {} and {}',

        'monitorenter': 'enter monitor for object {}',
        'monitorexit': 'exit monitor for object {}',

        'nop': 'do nothing',

        'wide': '',
    }
    return instruction2tran


def control_Category():
    category = {
        'uncond': [
            'goto', 'goto_w'
        ],
        'cond': [
            'ifeq', 'iflt', 'ifle', 'ifgt', 'ifge',
            'ifnull', 'ifnonnull',
            'if_icmpeq', 'if_icmpgt', 'if_icmpge', 'if_acmpeq',
            'if_icmplt', 'if_icmple', 'if_icmpne', 'if_acmpne',
        ],
        'compound_cond': [
            'tableswitch', 'lookupswitch',
        ],
        'comparisons': [
            'lcmp', 'fcmpg', 'fcmpl', 'dcmpg', 'dcmpl',
        ]
    }

    return category


def operate_Category():
    category = {
        'cal_0_goto_branch_category': [
            'goto', 'goto_w',
            'jsr', 'jsr_w',
        ],
        'cal_1_goto_branch_category': [
            'ifeq', 'ifne', 'iflt', 'ifge', 'ifgt', 'ifle',
            'ifnonnull', 'ifnull',
            'lookupswitch', 'tableswitch',
        ],
        'cal_2_goto_branch_category': [
            'if_acmpeq', 'if_acmpne',
            'if_icmpeq', 'if_icmpne', 'if_icmplt', 'if_icmpge', 'if_icmpgt', 'if_icmple',
        ],
        'cal_0_return_category': [
            'ret', 'return',
        ],
        'cal_1_return_category': [
            'areturn', 'dreturn', 'freturn', 'ireturn', 'lreturn',
        ],
        'cal_0_push_stack_category': [
            'aload', 'aload_0', 'aload_1', 'aload_2', 'aload_3',
            'dload', 'dload_0', 'dload_1', 'dload_2', 'dload_3',
            'fload', 'fload_0', 'fload_1', 'fload_2', 'fload_3',
            'iload', 'iload_0', 'iload_1', 'iload_2', 'iload_3',
            'lload', 'lload_0', 'lload_1', 'lload_2', 'lload_3',
            'aconst_null',
            'dconst_0', 'dconst_1',
            'fconst_0', 'fconst_1', 'fconst_2',
            'iconst_m1', 'iconst_0', 'iconst_1', 'iconst_2', 'iconst_3', 'iconst_4', 'iconst_5',
            'lconst_0', 'lconst_1',
            'bipush', 'sipush',
            'ldc', 'ldc_w', 'ldc2_w',
            'getstatic',
        ],

        'cal_1_push_stack_category': [
            'd2f', 'd2i', 'd2l',
            'f2d', 'f2i', 'f2l',
            'i2b', 'i2c', 'i2d', 'i2f', 'i2l', 'i2s',
            'l2d', 'l2f', 'l2i',
            'dneg', 'fneg', 'ineg', 'lneg',
            'instanceof',
            'checkcast',
            'arraylength',
            'athrow',
            'getfield',
        ],
        'cal_2_push_stack_category': [
            'aaload', 'baload', 'caload', 'daload', 'faload', 'iaload', 'laload', 'saload',
            'dadd', 'ddiv', 'dmul', 'drem', 'dsub',
            'fadd', 'fdiv', 'fmul', 'frem', 'fsub',
            'iadd', 'idiv', 'imul', 'irem', 'isub',
            'ladd', 'ldiv', 'lmul', 'lrem', 'lsub',
            'iand', 'ior', 'ixor',
            'ishl', 'ishr', 'iushr',
            'land', 'lor', 'lxor',
            'lshl', 'lshr', 'lushr',
            'dcmpg', 'dcmpl', 'fcmpg', 'fcmpl', 'lcmp',
            'swap',

        ],
        'cal_3_push_stack_category': [

        ],
        'cal_n_push_stack_category': [
            'invokedynamic', 'invokeinterface', 'invokespecial', 'invokestatic', 'invokevirtual',
        ],
        'cal_0_store_var_category': [
            'iinc', 'iinc_w'
        ],
        'cal_1_store_var_category': [
            'astore', 'astore_0', 'astore_1', 'astore_2', 'astore_3',
            'dstore', 'dstore_0', 'dstore_1', 'dstore_2', 'dstore_3',
            'fstore', 'fstore_0', 'fstore_1', 'fstore_2', 'fstore_3',
            'istore', 'istore_0', 'istore_1', 'istore_2', 'istore_3',
            'lstore', 'lstore_0', 'lstore_1', 'lstore_2', 'lstore_3',
            'putstatic',
        ],
        'cal_2_store_var_category': [
            'putfield',
        ],
        'cal_3_store_var_category': [
            'aastore', 'bastore', 'castore', 'dastore', 'fastore', 'iastore', 'lastore', 'sastore',
        ],
        'cal_0_create_category': [
            'new',
        ],
        'cal_1_create_category': [
            'anewarray', 'newarray',
        ],
        'cal_n_create_category': [
            'multianewarray',
        ],
        'management_stack_category': [
            'pop', 'pop2', 'dup', 'dup_x1', 'dup_x2', 'dup2', 'dup2_x1', 'dup2_x2',
        ],
        'cal_1_monitor_category': [
            'monitorenter', 'monitorexit',
        ],
        'do_nothing_category': [
            'nop',
        ],
        'unknow_category': [
            'wide',
        ],
    }

    return category
