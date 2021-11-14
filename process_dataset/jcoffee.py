import subprocess
import re
import os
import sys
import shutil
import string
from collections import defaultdict
import time

CLASS_NAME_STR = "public class"
MY_GLOBAL_HELPER_CLASS = "UNKNOWN"
SEPARATOR_STRING = "// Code below this line has been added to remove errors"
PUBLIC_CLASS_STRING = "// This class was public"

DUMMY_RETURN_TYPE = MY_GLOBAL_HELPER_CLASS

LIST_OF_INTERNAL_EXCEPTIONS = ["UncheckedIOException", "ArithmeticException", "ArrayIndexOutOfBoundsException",
                               "ArrayStoreException", "ClassCastException", "IllegalArgumentException",
                               "IllegalMonitorStateException", "IllegalStateException", "IllegalThreadStateException",
                               "IndexOutOfBoundsException", "NegativeArraySizeException", "NullPointerException",
                               "NumberFormatException", "SecurityException", "UnsupportedOperationException"]
NATIVE_ARRAY_VARIABLES = ["length"]
RESERVED_KEYWORDS = ["super"]
VALID_VARIABLE_CHARS = string.ascii_letters + string.digits + "_"
VALID_CHARS = VALID_VARIABLE_CHARS + "$.()[]"
IDENTIFIERS_FOR_FUNCTION = ["public", "private", "protected"]

l_code = []
dont_touch_list = set()
d_classes_to_add = defaultdict()
reverse_method_to_class_mapping = defaultdict()
exception_list_index = 0


def compile_file_and_get_output(file_path, class_dir):
    return subprocess.run(['javac', file_path, '-d', class_dir], stderr=subprocess.PIPE).stderr.decode('utf-8')


def get_file_content(file_path):
    content = None
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def remove_file_if_exists(java_dir, class_dir):
    try:
        shutil.rmtree(java_dir)
        shutil.rmtree(class_dir)
    except OSError:
        pass


# 将duplicate_file修改为make_dir
# 将源代码中的使用命令新建文件夹和复制文件改为使用代码的形式
def make_dir(new_content, file_path, class_dir):
    # 创建java存放的临时文件夹
    if (file_path.rfind('/') != -1):
        dir_path = file_path[:file_path.rfind('/')]
        os.makedirs(dir_path, exist_ok=True)
    # 创建class存放的临时文件夹
    os.makedirs(class_dir, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)


def write_file_content(file_path, file_content):
    if (file_path.rfind('/') != -1):
        dir_path = file_path[:file_path.rfind('/')]
        os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(file_content)


def get_full_path_of_files_in_folder(folder_path):
    l_docs = []
    paths = os.listdir(folder_path)
    for i in range(len(paths)):
        paths[i] = folder_path + paths[i]
        if (os.path.isdir(paths[i])):
            l_docs.extend(get_full_path_of_files_in_folder(paths[i] + "/"))
        else:
            l_docs.append(paths[i])
    return l_docs


""" Functions to edit code """


def get_l_of_params(s):
    if (s == ""):
        return []
    s = s.replace("<null>", "Object")
    s = re.sub('Class<[^>]*>', 'Class', s)
    l_p = s.split(",")
    to_ret = []
    pending = []
    for i in range(len(l_p)):
        if ("<" in l_p[i] or ">" in l_p[i]):
            pending.append(l_p[i])
        else:
            if (pending != []):
                to_ret.append(",".join(pending))
                pending = []
            to_ret.append(l_p[i])

    if (pending != []):
        to_ret.append(",".join(pending))
    return to_ret


def make_dummy_method(method_sig):
    ret_type = method_sig[0]

    method_name = method_sig[1].split("(")[0]
    method_param_types = get_l_of_params(method_sig[1].split("(")[1][:-1])
    method_args = []
    for ctr in range(len(method_param_types)):
        method_args.append(method_param_types[ctr] + " o" + str(ctr))

    method_args = "(" + ", ".join(method_args) + ")"
    method_definition = '\tpublic ' + ret_type + ' ' + method_name + method_args + '{ return null; }\n'

    return method_definition


def make_dummy_variable(var):
    if (type(var) != type('') and len(var) == 2):
        return '\tpublic ' + var[0] + ' ' + var[1] + ";\n"
    else:
        return '\tpublic ' + DUMMY_RETURN_TYPE + ' ' + var + ";\n"


def make_dummy_constructor(class_name, cons_arg):
    cons_param_types = get_l_of_params(cons_arg)
    cons_params = []
    for ctr in range(len(cons_param_types)):
        cons_params.append(cons_param_types[ctr] + " o" + str(ctr))
    cons_params = "(" + ", ".join(cons_params) + ")"
    return '\t' + class_name + cons_params + '{}\n'


def make_dummy_class(class_name, list_of_variables, list_of_method_signatures, list_of_constructor_args):
    variable_definitions = ""
    method_definitions = ""

    for var in list_of_variables:
        variable_definitions += make_dummy_variable(var)

    for cons_arg in list_of_constructor_args:
        method_definitions += make_dummy_constructor(class_name, cons_arg)

    for method_sig in list_of_method_signatures:
        method_definitions += make_dummy_method(method_sig)

    method_definitions = method_definitions.strip()
    variable_definitions = variable_definitions.strip()
    class_code = "class " + class_name + " {\n\t" + variable_definitions + "\n\t" + method_definitions + "\n}\n"
    return class_code


def make_dummy_exception(class_name, list_of_constructor_args):
    method_definitions = "\tpublic " + class_name + "(String errorMessage) { super(errorMessage); }\n"
    for cons_arg in list_of_constructor_args:
        method_definitions += make_dummy_constructor(class_name, cons_arg)

    class_code = "class " + class_name + " extends Exception{\n" + method_definitions + "}\n"
    return class_code


def get_existing_class_names():
    global l_code

    contenders = []
    for line_no, line in enumerate(l_code):
        if (SEPARATOR_STRING in line):
            break
        if (line.startswith("public class ")):
            contenders.append((line_no, line.split(" ")[2]))
        if (line.startswith("final class ")):
            contenders.append((line_no, line.split(" ")[2]))
        if (line.startswith("class ")):
            contenders.append((line_no, line.split(" ")[1]))

        if (line.strip().startswith("@")):
            l_code[line_no] = "//" + l_code[line_no]
    return contenders


def get_code_for_new_class(class_name):
    l_var = [i[1] if len(i) == 2 else i[1:] for i in d_classes_to_add[class_name] if i[0] == "VAR"]
    l_method = [i[1] if len(i) == 2 else i[1:] for i in d_classes_to_add[class_name] if i[0] == "METHOD"]
    l_exception = [i[1] for i in d_classes_to_add[class_name] if i[0] == "EXCEPTION"]
    l_constructor = [i[1] for i in d_classes_to_add[class_name] if i[0] == "CONSTRUCTOR"]

    if (len(l_exception) != 0):
        class_code = make_dummy_exception(class_name, l_constructor)
    else:
        class_code = make_dummy_class(class_name, l_var, l_method, l_constructor)

    return class_code.split("\n")


def get_code_for_existing_class(class_name):
    l_var = [i[1] if len(i) == 2 else i[1:] for i in d_classes_to_add[class_name] if i[0] == "VAR"]
    l_method = [i[1] if len(i) == 2 else i[1:] for i in d_classes_to_add[class_name] if i[0] == "METHOD"]

    method_definitions = ""
    variable_definitions = ""
    for method_sig in l_method:
        method_definitions += make_dummy_method(method_sig)
    for variable_sig in l_var:
        variable_definitions += make_dummy_variable(variable_sig)

    class_code = variable_definitions + method_definitions
    return class_code


def add_members_of_existing_existing_class():
    global l_code
    global d_classes_to_add
    global existing_class_names

    existing_class_names = get_existing_class_names()

    for line_no, class_name in existing_class_names:
        class_code = get_code_for_existing_class(class_name).strip().split("\n")
        if (class_code != ['']):
            l_code = l_code[:line_no + 1] + class_code + l_code[line_no + 1:]
        # reset dictionary for existing classes
        d_classes_to_add[class_name] = set()


def get_new_code_to_add():
    new_code_to_add = []

    for class_name in d_classes_to_add.keys():
        if (class_name not in [i[1] for i in existing_class_names]):
            new_code_to_add += get_code_for_new_class(class_name)

    return new_code_to_add


def get_new_code_line_after_add_data_type(pointer, valid_chars, target_type, code_line, direction):
    target_type = "(" + target_type + ")" + "(Object)"
    if direction == "right":
        ptr = len(pointer) - 1

        while (code_line[ptr] not in valid_chars):
            ptr = ptr + 1
            if (ptr > len(code_line)):
                return code_line
        return code_line[:ptr] + target_type + code_line[ptr:]

    # direction = "left"
    ptr = len(pointer) - 2
    num_brackets = 0
    num_sq_brackets = 0

    if (code_line[:ptr + 1].strip() in ["switch"]):
        ptr += 1

    if (code_line[ptr + 1] not in valid_chars):
        while (code_line[ptr] == " "):
            ptr -= 1
            if (ptr < 0):
                return code_line

    while (num_sq_brackets != 0 or num_brackets != 0 or code_line[ptr] in valid_chars):
        if (code_line[ptr] == "]"):
            num_sq_brackets += 1
        elif (code_line[ptr] == "[" and num_sq_brackets != 0):
            num_sq_brackets -= 1

        elif (code_line[ptr] == ")"):
            num_brackets += 1
        elif (code_line[ptr] == "(" and num_brackets != 0):
            num_brackets -= 1

        elif (code_line[ptr] in "[(" and num_brackets == 0 and num_sq_brackets == 0):
            break
        ptr -= 1

        if (ptr < 0):
            return code_line

    if (code_line[ptr] != " "):
        ptr += 1

    if (code_line[ptr:].strip().startswith("return")):
        while (code_line[ptr] == " "):
            ptr += 1
        ptr += len("return ")

    if (code_line[ptr - 3:ptr] == "new"):
        # code_line = code_line[:ptr-3] + "(" + code_line[ptr-3:-1] + ")" + code_line[-1]
        ptr -= 3

    return code_line[:ptr] + target_type + code_line[ptr:]


def compare_args_by_index(sig_src, sig_dst):
    l_src = sig_src[1:-1].split(",")
    l_dst = sig_dst[1:-1].split(",")

    for i in range(len(l_src)):
        if (l_src[i] != l_dst[i]):
            return i, l_dst[i]


def get_new_code_line_after_add_suitable_method_signature(pointer, method_desc, code_line, possible_matches):
    method_name = method_desc.split("(")[0].strip()
    method_args = method_desc.strip()[len(method_name):]

    # always apply the match - argument mismatch; ... cannot be converted to ...
    match_to_apply = ""
    for i in possible_matches:
        if (i[1].startswith("(argument mismatch;")):
            match_to_apply = i[0][i[0].find("("): i[0].find(")") + 1]
            break

    if (match_to_apply == ""):  # no match found - make new method/constructor otherwise
        class_name = possible_matches[0][0].split(" ")[1].split(".")[0]
        if (possible_matches[0][0].startswith("method")):
            reverse_method_to_class_mapping[method_desc] = class_name
            d_classes_to_add[class_name].add(("METHOD", DUMMY_RETURN_TYPE, method_desc))
        else:
            d_classes_to_add[class_name].add(("CONSTRUCTOR", method_desc.split("(")[1][:-1]))
        return code_line

    index_to_change, target_type = compare_args_by_index(method_args, match_to_apply)
    target_type = "(" + target_type + ")" + "(Object)"

    ptr = len(pointer) - 2
    while (not (
            code_line[ptr:].startswith(method_name) or code_line[ptr:].startswith("this") or code_line[ptr:].startswith(
        "super"))):
        ptr += 1
        if (ptr > len(code_line)):
            return code_line

    num_brackets = num_comma = 0

    while (not (num_brackets == 1 and num_comma == index_to_change)):
        if (code_line[ptr] == ")"):
            num_brackets -= 1
        elif (code_line[ptr] == "("):
            num_brackets += 1
        if (num_brackets == 1 and code_line[ptr] == ","):
            num_comma += 1
        ptr += 1

        if (ptr > len(code_line)):
            return code_line

    code_line = code_line[:ptr] + target_type + code_line[ptr:]
    return code_line


def ls_does_contain(ls, subs):
    for s in ls:
        if (subs in s):
            return True
    return False


def b_begins_with_one_of_a(a, b):
    for i in a:
        if (b.startswith(i)):
            return True
    return False


# adds "throws Throwable" to the first function in class
def make_existing_function_throwable():
    global l_code

    line_number = 0
    while line_number < len(l_code):
        if (b_begins_with_one_of_a(IDENTIFIERS_FOR_FUNCTION, l_code[line_number].strip())):
            break
        line_number += 1

    if (line_number == len(l_code)):
        return

    if ("throws " in l_code[line_number]):
        if ("throws Throwable" in l_code[line_number]):
            return
        l_code[line_number] = l_code[line_number].replace("throws ", "throws Throwable, ")
        return

    if (l_code[line_number].strip()[-1] == "{"):
        l_code[line_number] = l_code[line_number].rstrip()[:-1] + " throws Throwable {"

    elif (l_code[line_number].strip()[-1] == ")"):
        l_code[line_number] = l_code[line_number].rstrip() + " throws Throwable "

    return


class Exception:

    def __init__(self, e):

        self.l_e = [i for i in e.split('\n')]
        self.line_number = int(self.l_e[0].strip()[:self.l_e[0].find(":")]) - 1
        self.exception_desc = self.l_e[0].strip()[self.l_e[0].find("error: ") + 7:]
        if (self.can_touch_this()):
            if ("error: " in self.l_e[0]):
                self.analyse_exception()

    def can_touch_this(self):
        return self.line_number not in dont_touch_list

    def add_to_dont_touch_list(self):
        global dont_touch_list
        dont_touch_list.add(self.line_number)

    def analyse_exception(self):
        global l_code
        global d_classes_to_add
        global reverse_method_to_class_mapping
        global exception_list_index

        # class ... is public, should be declared in a file named ....java
        if ("is public, should be declared in a file named" in self.exception_desc):
            l_code[self.line_number] = PUBLIC_CLASS_STRING + "\n" + l_code[self.line_number].replace("public class",
                                                                                                     "class")
            self.add_to_dont_touch_list()
            return

        # package .... does not exist
        if (self.exception_desc.startswith("package ") and self.exception_desc.endswith("does not exist")):
            package_name = self.exception_desc.split(" ")[1]
            if (l_code[self.line_number].startswith("import ")):
                l_code[self.line_number] = "// " + l_code[self.line_number]
            else:
                l_code[self.line_number] = l_code[self.line_number].replace(package_name + ".", "")
            self.add_to_dont_touch_list()
            return

        # call to super must be first statement in constructor
        # Action: rename method to class_name
        if ("call to super must be first statement in constructor" in self.exception_desc):
            prev_line = l_code[self.line_number - 1]
            if (prev_line.count("(") != 1):
                return
            prev_line = prev_line.replace(" void ", " ")
            bracket_index = prev_line.find("(")
            method_start_index = prev_line[:bracket_index].rfind(' ') + 1
            class_name = get_existing_class_names()[0][1]
            l_code[self.line_number - 1] = prev_line[:method_start_index] + class_name + prev_line[bracket_index:]
            self.add_to_dont_touch_list()
            return

        # incorrect constructor
        if ("cannot be applied to given types;" in self.exception_desc and "constructor" in self.exception_desc):
            class_name = self.exception_desc.split(" ")[1]
            needed_args_line = [i.strip() for i in self.l_e if i.strip().startswith("found:")][0]
            needed_args = needed_args_line.split(":")[1].strip()

            if (needed_args == "no arguments"):
                needed_args = ""
            if ("..." in needed_args):
                return
            d_classes_to_add[class_name].add(("CONSTRUCTOR", needed_args))
            d_classes_to_add[class_name].add(("CONSTRUCTOR", ""))
            return

        # incorrect method
        if ("cannot be applied to given types;" in self.exception_desc and "method" in self.exception_desc):
            method_name = self.exception_desc.split(" ")[1]
            class_name = self.exception_desc.split(" ")[4]
            needed_args_line = [i.strip() for i in self.l_e if i.strip().startswith("found:")][0]
            needed_args = needed_args_line.split(":")[1].strip()

            if (needed_args == "no arguments"):
                needed_args = ""
            if ("..." in needed_args):
                return
            d_classes_to_add[class_name].add(("METHOD", DUMMY_RETURN_TYPE, method_name + "(" + needed_args + ")"))
            reverse_method_to_class_mapping[method_name + "(" + needed_args + ")"] = class_name
            return

        # variable/method xyz is already defined in
        # Action: commment out this code_statement
        if ((self.exception_desc.startswith("variable") or self.exception_desc.startswith(
                "method")) and "is already defined in " in self.exception_desc):
            l_code[self.line_number] = "//" + l_code[self.line_number]

            self.add_to_dont_touch_list()
            return

        # for-each not applicable to expression type
        if (self.exception_desc.startswith("for-each not applicable to expression type")):
            ptr = len(self.l_e[2]) - 2

            while (l_code[self.line_number][ptr] in VALID_CHARS):
                ptr -= 1
                if (ptr < 0):
                    return
            ptr += 1
            new_code = l_code[self.line_number]
            l_code[self.line_number] = new_code[:ptr] + "(Object[])(Object)" + new_code[ptr:]

            self.add_to_dont_touch_list()
            return

        # incompatible types: ... cannot be converted to ...
        if (self.exception_desc.startswith("incompatible types: ") and "cannot be converted to" in self.exception_desc):
            source_type = self.exception_desc.split(" ")[-6]
            target_type = self.exception_desc.split(" ")[-1]

            if (source_type == "Object"):
                target_type = target_type + "[]"

            if (target_type == DUMMY_RETURN_TYPE):
                var_name = l_code[self.line_number - 1].strip().split(" ")
                if (len(var_name) < 2):
                    l_code[self.line_number] = get_new_code_line_after_add_data_type(self.l_e[2], VALID_CHARS,
                                                                                     target_type,
                                                                                     l_code[self.line_number], "left")
                    self.add_to_dont_touch_list()
                    return
                var_name = var_name[1]
                partial_creation_str = target_type + " " + var_name + " = new " + target_type + "();"
                if (partial_creation_str not in l_code[self.line_number - 1].strip()):
                    l_code[self.line_number] = get_new_code_line_after_add_data_type(self.l_e[2], VALID_CHARS,
                                                                                     target_type,
                                                                                     l_code[self.line_number], "left")
                    self.add_to_dont_touch_list()
                    return

                if ((var_name + ".") in l_code[self.line_number]):
                    class_name = target_type
                    s_index = l_code[self.line_number].find(var_name + ".") + len(var_name + ".")
                    e_index = l_code[self.line_number][s_index:].find(" ")
                    new_var_name = l_code[self.line_number][s_index:s_index + e_index]

                    d_classes_to_add[class_name].remove(("VAR", new_var_name))
                    d_classes_to_add[class_name].add(("VAR", source_type, new_var_name))

                else:
                    l_code[self.line_number - 1] = l_code[self.line_number - 1].replace(target_type, source_type)
                    l_code[self.line_number - 1] = l_code[self.line_number - 1].replace(" = ", " = NULL; //")

                self.add_to_dont_touch_list()
                return

            # if(source_type == "Object"):
            # 	target_type = target_type + "[]"

            direction = "left"

            if (target_type == "Throwable"):
                target_type = "(" + target_type + ")(Object)"
                curr_line = l_code[self.line_number]
                insert_pos = curr_line.find(" ", len(self.l_e[2]))
                new_line = curr_line[:insert_pos] + target_type + curr_line[insert_pos:]
                l_code[self.line_number] = new_line

            elif (l_code[self.line_number][len(self.l_e[2]) - 1] == "?"):
                ptr_pos = len(self.l_e[2]) - 1
                target_type = "(" + target_type + ")(Object)"
                curr_line = l_code[self.line_number]
                colon_pos = curr_line.find(":", ptr_pos)
                new_line = curr_line[:ptr_pos + 1] + target_type + curr_line[
                                                                   ptr_pos + 1:colon_pos + 1] + target_type + curr_line[
                                                                                                              colon_pos + 1:]
                l_code[self.line_number] = new_line
            else:
                l_code[self.line_number] = get_new_code_line_after_add_data_type(self.l_e[2], VALID_CHARS, target_type,
                                                                                 l_code[self.line_number], direction)
            self.add_to_dont_touch_list()
            return

        # bad operand types for binary operator '..'
        if (self.exception_desc.startswith("bad operand types for binary operator")):
            first_type_line = [i.strip() for i in self.l_e if i.strip().startswith("first type:")][0]
            second_type_line = [i.strip() for i in self.l_e if i.strip().startswith("second type:")][0]

            first_type = first_type_line.split(":")[1].strip()
            second_type = second_type_line.split(":")[1].strip()

            target_type = second_type
            direction = "left"

            if (second_type == first_type):
                direction = "right"
                if (self.exception_desc.split(" ")[-1] in ["'&&'", "'||'"]):
                    target_type = "Boolean"
                else:
                    target_type = "int"

            elif (second_type == DUMMY_RETURN_TYPE):
                target_type = first_type
                direction = "right"

            l_code[self.line_number] = get_new_code_line_after_add_data_type(self.l_e[2], VALID_CHARS, target_type,
                                                                             l_code[self.line_number], direction)

            self.add_to_dont_touch_list()
            return

        # incomparable types: Object and int
        if (self.exception_desc.startswith("incomparable types: ")):
            l_words = self.exception_desc.split(" ")
            type_a = l_words[2]
            type_b = l_words[4]

            target_type = type_a
            direction = "right"
            if (type_a == DUMMY_RETURN_TYPE):
                target_type = type_b
                direction = "left"

            l_code[self.line_number] = get_new_code_line_after_add_data_type(self.l_e[2], VALID_CHARS, target_type,
                                                                             l_code[self.line_number], direction)

            self.add_to_dont_touch_list()
            return

        # interface expected here
        if (self.exception_desc.startswith("interface expected here")):
            start_ind = l_code[self.line_number].find("implements")
            end_ind = len(l_code[self.line_number])
            if (l_code[self.line_number].endswith("{")):
                end_ind -= 1
            l_code[self.line_number] = l_code[self.line_number][:start_ind] + l_code[self.line_number][end_ind:]
            self.add_to_dont_touch_list()
            return

        if ("invalid method declaration; return type required" in self.exception_desc):
            ptr = len(self.l_e[2]) - 1
            new_line = l_code[self.line_number][:ptr] + " void " + l_code[self.line_number][ptr:]
            l_code[self.line_number] = new_line
            self.add_to_dont_touch_list()
            return

        # exception ABCException is never thrown in body of corresponding try statement
        if ("is never thrown in body of corresponding try statement" in self.exception_desc):
            exception_name = self.exception_desc.split(" ")[1]
            l_code[self.line_number] = l_code[self.line_number].replace(exception_name, LIST_OF_INTERNAL_EXCEPTIONS[
                exception_list_index + 1])
            exception_list_index += 1
            self.add_to_dont_touch_list()
            return

        # exception ... has already been caught
        if ("exception" in self.exception_desc and " has already been caught" in self.exception_desc):
            exception_name = self.exception_desc.split(" ")[1]
            if (exception_name in LIST_OF_INTERNAL_EXCEPTIONS):
                new_exception = LIST_OF_INTERNAL_EXCEPTIONS[
                    (LIST_OF_INTERNAL_EXCEPTIONS.index(exception_name) + 1) % len(LIST_OF_INTERNAL_EXCEPTIONS)]
            else:
                new_exception = LIST_OF_INTERNAL_EXCEPTIONS[0]
            l_code[self.line_number] = l_code[self.line_number].replace(exception_name, new_exception)
            self.add_to_dont_touch_list()
            return

        # no suitable method/constructor found for xyz(UNKNOWN)
        if (self.exception_desc.startswith("no suitable method found for") or self.exception_desc.startswith(
                "no suitable constructor found for")):
            method_desc = self.exception_desc.split(" ")[-1]

            possible_matches = []
            for i in range(3, len(self.l_e), 2):
                if ("not applicable" not in self.l_e[i].strip()):
                    break
                possible_matches.append((self.l_e[i].strip(), self.l_e[i + 1].strip()))

            l_code[self.line_number] = get_new_code_line_after_add_suitable_method_signature(self.l_e[2], method_desc,
                                                                                             l_code[self.line_number],
                                                                                             possible_matches)
            self.add_to_dont_touch_list()
            return

        # array required, but ... found
        if (self.exception_desc.startswith("array required, but ")):
            pointer = self.l_e[2]
            ptr = len(pointer) - 2

            var_name_start_ptr = ptr
            while (l_code[self.line_number][var_name_start_ptr] in VALID_VARIABLE_CHARS):
                var_name_start_ptr -= 1

            var_name_start_ptr += 1
            host_var = l_code[self.line_number][var_name_start_ptr:ptr + 1]

            host_class = reverse_method_to_class_mapping[host_var]
            if (("VAR", host_var) in d_classes_to_add[host_class]):
                d_classes_to_add[host_class].remove(("VAR", host_var))
            d_classes_to_add[host_class].add(("VAR", DUMMY_RETURN_TYPE + "[]", host_var))

            creation_str = DUMMY_RETURN_TYPE + " " + host_var + " = new " + DUMMY_RETURN_TYPE + "();"
            if (l_code[self.line_number - 1].strip() == creation_str):
                l_code[
                    self.line_number - 1] = DUMMY_RETURN_TYPE + "[] " + host_var + " = new " + DUMMY_RETURN_TYPE + "[5];"

            return

        # bad operand type ... for unary operator '++'
        if (self.exception_desc.startswith("bad operand type ") and " for unary operator " in self.exception_desc):

            if (self.exception_desc.endswith("'!'")):  # to go right
                ptr = len(self.l_e[2])
                l_code[self.line_number] = l_code[self.line_number][:ptr] + "(Boolean)(Object)" + l_code[
                                                                                                      self.line_number][
                                                                                                  ptr:]
                self.add_to_dont_touch_list()
                return

            pointer = self.l_e[2]
            ptr = len(pointer) - 2

            while (l_code[self.line_number][ptr] not in VALID_VARIABLE_CHARS):
                ptr -= 1
            var_name_start_ptr = ptr
            while (l_code[self.line_number][var_name_start_ptr] in VALID_VARIABLE_CHARS):
                var_name_start_ptr -= 1

            var_name_start_ptr += 1
            host_var = l_code[self.line_number][var_name_start_ptr:ptr + 1]

            host_class = reverse_method_to_class_mapping[host_var]
            if (("VAR", host_var) in d_classes_to_add[host_class]):
                d_classes_to_add[host_class].remove(("VAR", host_var))
            d_classes_to_add[host_class].add(("VAR", "int", host_var))

            return

        # ABC is already defined in this compilation unit
        # Action: Comment out this line
        if (" is already defined in this compilation unit" in self.exception_desc):
            l_code[self.line_number] = "// " + l_code[self.line_number]
            self.add_to_dont_touch_list()
            return

        # non-static variable ... cannot be referenced from a static context
        if (self.exception_desc.startswith(
                "non-static variable") and " cannot be referenced from a static context" in self.exception_desc):
            host_var = self.exception_desc.split()[2]
            host_class = reverse_method_to_class_mapping[host_var]
            if (("VAR", host_var) in d_classes_to_add[host_class]):
                d_classes_to_add[host_class].remove(("VAR", host_var))

            d_classes_to_add[host_class].add(("VAR", "static " + DUMMY_RETURN_TYPE, host_var))
            return

        # non-static method ... cannot be referenced from a static context
        if (self.exception_desc.startswith(
                "non-static method") and " cannot be referenced from a static context" in self.exception_desc):
            host_method = self.exception_desc.split()[2]
            host_class = reverse_method_to_class_mapping[host_method]

            if (("METHOD", DUMMY_RETURN_TYPE, host_method) in d_classes_to_add[host_class]):
                d_classes_to_add[host_class].remove(("METHOD", DUMMY_RETURN_TYPE, host_method))

            d_classes_to_add[host_class].add(("METHOD", "static " + DUMMY_RETURN_TYPE, host_method))
            return

        # type ABC does not take parameters
        # ABC<QQQ> --> ABC
        if (self.exception_desc.startswith("type ") and "does not take parameters" in self.exception_desc):
            ptr = len(self.l_e[2]) - 2
            l_code[self.line_number] = l_code[self.line_number][:ptr] + re.sub(r'\<[^>]*\>', '',
                                                                               l_code[self.line_number][ptr:])
            self.add_to_dont_touch_list()
            return

        # ABC is abstract; cannot be instantiated
        if ("is abstract; cannot be instantiated" in self.exception_desc):
            class_name = self.exception_desc.split(" ")[0]
            d_classes_to_add[class_name] = set()

            self.add_to_dont_touch_list()
            return

        # cannot find symbol
        if (self.exception_desc.startswith("cannot find symbol")):
            symbol_line = [i.strip() for i in self.l_e if i.strip().startswith("symbol:")][0]
            symbol = symbol_line.split(":")[1].strip()
            if ("..." in symbol):
                return

            if (not ls_does_contain(self.l_e, "location")):  # no location line

                # symbol: class ABC
                # Action: create new class ABC
                if (symbol.startswith("class ")):
                    class_name = symbol.split(" ")[1].strip()
                    if (class_name.endswith("Exception") or class_name.startswith("error") or (
                            "catch" in l_code[self.line_number])):
                        d_classes_to_add[class_name].add(("EXCEPTION", class_name))
                    if (class_name not in d_classes_to_add.keys()):
                        d_classes_to_add[class_name] = set()

                # symbol: method xyz()
                # Action: create new method xyz() in all existing classes
                if (symbol.startswith("method ")):

                    # Action: this is mp a "super" call. Add method to corresponding class
                    if ("super." in l_code[self.line_number]):  # super.setValue(String) --> this.setValue(String)
                        l_code[self.line_number] = l_code[self.line_number].replace("super.", "this.")
                        self.add_to_dont_touch_list()
                        return

                    method_desc = symbol.split(" ")[1].strip()
                    method_desc = method_desc.replace("<null>", "Object")

                    for line_no, class_name in get_existing_class_names():
                        d_classes_to_add[class_name].add(("METHOD", DUMMY_RETURN_TYPE, method_desc))
                        reverse_method_to_class_mapping[method_desc] = class_name
                    return

                # symbol: variable xyz
                # Action: create new instance xyz of type DUMMY
                if (symbol.startswith("variable ")):
                    var_name = symbol.split(" ")[1].strip()
                    class_name = MY_GLOBAL_HELPER_CLASS

                    # a reference to self variable
                    if (self.l_e[1].find("this." + var_name) != -1):
                        for line_no, class_name in get_existing_class_names():
                            d_classes_to_add[class_name].add(("VAR", var_name))
                    else:
                        spaces = " " * (len(l_code[self.line_number]) - len(l_code[self.line_number].lstrip()))
                        is_static = " static " * int(" static " in l_code[self.line_number])
                        new_line = spaces + is_static + class_name + " " + var_name + " = new " + class_name + "();"
                        l_code[self.line_number] = new_line + "\n" + l_code[self.line_number]
                        reverse_method_to_class_mapping[var_name] = class_name
                    self.add_to_dont_touch_list()
                return

            location_line = [i.strip() for i in self.l_e if i.strip().startswith("location:")][0]
            location = location_line.split(":")[1].strip()

            # symbol: class ABC
            # location: package a.b.c
            # Action: comment out the line
            if (symbol.startswith("class ") and location.startswith("package ")):
                l_code[self.line_number] = "// " + l_code[self.line_number]
                self.add_to_dont_touch_list()

                return

            # symbol: class ABC
            # location: class XYZ
            # Action: create new class ABC
            if (symbol.startswith("class ") and location.startswith("class ")):
                class_name = symbol.split(" ")[1].strip()
                if (class_name.endswith("Exception") or class_name.startswith("error")):
                    d_classes_to_add[class_name].add(("EXCEPTION", class_name))
                if (class_name not in d_classes_to_add.keys()):
                    d_classes_to_add[class_name] = set()
                return

            # symbol: method xyz(String)
            # location: class ABC
            # OR
            # location: variable e of type ABC
            # Action: create new method xyz(String) in class ABC
            if (symbol.startswith("method ") and (location.startswith("variable ") or location.startswith("class "))):
                method_desc = symbol.split(" ")[1].strip()
                class_name = location.split(" ")[-1].strip()

                method_desc = method_desc.replace("<null>", "Object")

                # Can remove this - we don not support functions as parameters
                method_desc = re.sub('Class<[^>]*>', 'Class', method_desc)
                method_desc = re.sub('<anonymous>', '', method_desc)

                d_classes_to_add[class_name].add(("METHOD", DUMMY_RETURN_TYPE, method_desc))
                reverse_method_to_class_mapping[method_desc] = class_name
                return

            # symbol: variable ...
            # location: variable e of type ABC
            # Action: create new variable ... in class ABC
            if (symbol.startswith("variable ") and location.startswith("variable ")):
                var_desc = symbol.split(" ")[1].strip()
                host_var = location.split(" ")[1].strip()
                class_name = location.split(" ")[-1].strip()

                if (var_desc in NATIVE_ARRAY_VARIABLES and class_name == DUMMY_RETURN_TYPE):
                    host_class = reverse_method_to_class_mapping[host_var]
                    d_classes_to_add[host_class].remove(("VAR", host_var))
                    d_classes_to_add[host_class].add(("VAR", DUMMY_RETURN_TYPE + "[]", host_var))
                    reverse_method_to_class_mapping[host_var] = host_class

                else:
                    d_classes_to_add[class_name].add(("VAR", var_desc))
                    reverse_method_to_class_mapping[var_desc] = class_name
                return

            # symbol: variable xyz
            # location: class ABC
            # Action: create new instance variable xyz of class ABC
            if (symbol.startswith("variable ") and location.startswith("class ")):
                var_name = symbol.split(" ")[1].strip()
                class_name = location.split(" ")[-1].strip()

                if (var_name in RESERVED_KEYWORDS):
                    return

                if (class_name in d_classes_to_add.keys()):
                    d_classes_to_add[class_name].add(("VAR", var_name))
                    reverse_method_to_class_mapping[var_name] = class_name
                    return

                class_name = MY_GLOBAL_HELPER_CLASS
                spaces = " " * (len(l_code[self.line_number]) - len(l_code[self.line_number].lstrip()))
                is_static = " static " * int(" static " in l_code[self.line_number])
                new_line = spaces + is_static + class_name + " " + var_name + " = new " + class_name + "();"
                l_code[self.line_number] = new_line + "\n" + l_code[self.line_number]
                reverse_method_to_class_mapping[var_name] = class_name
                self.add_to_dont_touch_list()

                return

            else:
                return

        else:
            return


def preprocess(original_code):
    new_content = ("import java.io.*;\n" +
                   "import java.lang.*;\n" +
                   "import java.util.*;\n" +
                   "import java.math.*;\n" +
                   "import java.net.*;\n" +
                   "import java.applet.*;\n" +
                   "import java.awt.*;\n" +
                   "import java.security.*;\n")

    old_content = original_code.strip()
    if (old_content.startswith("package ")):
        old_content = "//" + old_content
    new_content += old_content
    return new_content


def reset_d_classes():
    global d_classes_to_add
    global exception_list_index
    d_classes_to_add = defaultdict(set)
    d_classes_to_add[DUMMY_RETURN_TYPE] = set()
    d_classes_to_add[MY_GLOBAL_HELPER_CLASS] = set()
    reverse_method_to_class_mapping = defaultdict()
    exception_list_index = 0


def handle_file(original_code, code_name, java_dir, class_dir, times_to_try_compile):
    global l_code
    global dont_touch_list

    reset_d_classes()
    new_code = ''

    file_path = f'{java_dir}{code_name}.java'
    new_content = preprocess(original_code)
    make_dir(new_content, file_path, class_dir)

    l_code = new_content.split("\n")
    make_existing_function_throwable()

    i = 0
    while (i < times_to_try_compile):
        s = compile_file_and_get_output(file_path, class_dir)

        l_errors = [l.strip() for l in s.split(file_path + ":")][1:]
        l_errors = [i for i in l_errors if (" error: " in i and " warning: " not in i)]

        num_errors = len(l_errors)

        if (num_errors == 0):
            remove_file_if_exists(java_dir, class_dir)
            if new_code == '':
                new_code = new_content
            return new_code

        dont_touch_list = set()
        for j in range(len(l_errors)):
            e = Exception(l_errors[j])

        add_members_of_existing_existing_class()
        generated_code = get_new_code_to_add()
        new_code = "\n".join(l_code + [SEPARATOR_STRING] + generated_code)

        write_file_content(file_path, new_code)

        i += 1
    remove_file_if_exists(java_dir, class_dir)
    return None


def run(code_name, original_code):
    times_to_try_compile = 5

    java_dir = 'temp/java/'
    class_dir = 'temp/class/'

    try:
        jcoffee_code = handle_file(original_code, code_name, java_dir, class_dir, times_to_try_compile)
        if jcoffee_code:
            print(f'{code_name} jcoffee succeeded')
        else:
            print(f'{code_name} jcoffee failed')
    except:
        print(f"{code_name} has some error occured")
        return None

    remove_file_if_exists(java_dir, class_dir)
    return jcoffee_code

if __name__ == '__main__':
    path_ = ''
    code_file = open(path_, 'r', encoding='utf-8')
    code = code_file.read()
    jcode = run('name', code)
    print(jcode)