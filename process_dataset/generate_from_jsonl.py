from dpu_utils.utils import run_and_debug, RichPath, ChunkWriter
import argparse
from tqdm import tqdm
import pandas as pd
import jcoffee
import sys
import os


def get_start_end_idx(lines):
    list_ = []
    start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith('filename: ') and idx != 0:
            end_idx = idx
            list_.append([start_idx, end_idx])
            start_idx = idx
    list_.append([start_idx, len(lines)])

    return list_


def compilation_to_df(args):
    path_name = '00/'
    txt_dir = args.txt_path + path_name
    class_dir = args.class_path + path_name
    df_dir = args.df_path

    os.makedirs(df_dir, exist_ok=True)

    code_txt_file = open(txt_dir + 'code.txt', 'r', encoding='utf-8')
    doc_txt_file = open(txt_dir + 'doc.txt', 'r', encoding='utf-8')
    jcode_txt_file = open(txt_dir + 'jcode.txt', 'r', encoding='utf-8')

    c_lines = code_txt_file.readlines()
    d_lines = doc_txt_file.readlines()
    j_lines = jcode_txt_file.readlines()

    c_idx_list = get_start_end_idx(c_lines)
    d_idx_list = get_start_end_idx(d_lines)
    j_idx_list = get_start_end_idx(j_lines)

    code_txt_file.close()
    doc_txt_file.close()
    jcode_txt_file.close()

    assert len(c_idx_list) == len(d_idx_list) == len(j_idx_list)

    df_dict = {'name': [], 'code': [], 'doc': [], 'jcode': [], 'opcode': []}
    for i in range(len(c_idx_list)):
        name = c_lines[c_idx_list[i][0]][10:-1]
        code = ''.join(c_lines[c_idx_list[i][0] + 1:c_idx_list[i][1]])
        doc = ''.join(d_lines[d_idx_list[i][0] + 1:d_idx_list[i][1]])
        jcode = ''.join(j_lines[j_idx_list[i][0] + 1:j_idx_list[i][1]])

        df_dict['name'].append(name)
        df_dict['code'].append(code)
        df_dict['doc'].append(doc)
        df_dict['jcode'].append(jcode)

    names = df_dict['name']
    for name in tqdm(names):
        opcode_file = open(class_dir + name + '/' + name + '.txt', 'r', encoding='utf-8')
        opcode_str = opcode_file.read()
        df_dict['opcode'].append(opcode_str)
        opcode_file.close()

    df = pd.DataFrame(df_dict)
    df.to_pickle(df_dir + path_name[:-1] + '.pkl')


def compile_from_txt(args):
    path_name = '16/'
    compile_dirs = args.txt_path + path_name

    j_txt = open(compile_dirs + 'jcode.txt', 'r', encoding='utf-8')
    lines = j_txt.readlines()
    code_idx_list = []
    start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith('filename: ') and idx != 0:
            end_idx = idx
            code_idx_list.append([start_idx, end_idx])
            start_idx = idx

    code_idx_list.append([start_idx, len(lines)])
    print(f'code txt len: {len(code_idx_list)}')

    for idx_list in code_idx_list:
        start_ = idx_list[0]
        end_ = idx_list[1]

        filename = lines[start_][10:-1]
        print(filename)

        jcode_path = args.class_path + path_name + filename + '/'
        os.makedirs(jcode_path, exist_ok=True)
        jcode_file = open(jcode_path + filename + '.java', 'w', encoding='utf-8')
        jcode_str = ''.join(lines[start_ + 1:end_])
        jcode_file.write(jcode_str)
        jcode_file.close()

        os.system(f'javac -g {jcode_path}{filename}.java')
        os.system(f'javap -l -s -c -verbose {jcode_path}{filename} > {jcode_path}{filename}.txt')


def jcoffee_run(args):
    global c_idx

    def jcoffee_runner(df_):
        global c_idx
        func_name = df_['func_name'].split('.')[-1]
        file_name = f'{func_name}_{f_idx_str}{c_idx}'

        code_class = f'class {file_name} ' + '{\n' + df_['code'] + '\n}'
        jcoffee_code = jcoffee.run(file_name, code_class)

        if jcoffee_code:
            txt_code_file.write('filename: ' + file_name + '\n' + df_['code'] + '\n\n')
            txt_jcode_file.write('filename: ' + file_name + '\n' + jcoffee_code + '\n\n')
            txt_doc_file.write('filename: ' + file_name + '\n' + df_['docstring'] + '\n\n')
        c_idx += 1

    input_dir = args.jsonl_output_path
    files = os.listdir(input_dir)
    files = [i for i in files if i.endswith('.pkl')]

    for f in files:
        f_idx = int(f.split('.')[0].split('_')[-1])
        c_idx = 0
        f_idx_str = str(f_idx) if f_idx > 10 else f'0{str(f_idx)}'
        txt_dir = args.txt_path + f_idx_str + '/'
        if not os.path.exists(txt_dir):
            os.mkdir(txt_dir)

        txt_code_path = txt_dir + 'code.txt'
        txt_jcode_path = txt_dir + 'jcode.txt'
        txt_doc_path = txt_dir + 'doc.txt'

        txt_code_file = open(txt_code_path, 'w', encoding='utf8')
        txt_jcode_file = open(txt_jcode_path, 'w', encoding='utf8')
        txt_doc_file = open(txt_doc_path, 'w', encoding='utf8')

        input_path = f'{input_dir}{f}'
        df = pd.read_pickle(input_path)
        df.apply(jcoffee_runner, axis=1)
        c_idx = 0

        txt_code_file.close()
        txt_jcode_file.close()
        txt_doc_file.close()


def jsonl2data(args):
    azure_info_path = args.azure_info
    jsonl_input_path = RichPath.create(args.jsonl_input_path, azure_info_path)

    assert jsonl_input_path.is_dir(), 'Argument supplied must be a directory'
    files = list(jsonl_input_path.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    print(f'reading files from {jsonl_input_path.path}')
    for f in tqdm(files, total=len(files)):
        file_name = f.path.split('\\')[-1][:-9]
        df = pd.DataFrame(list(f.read_as_jsonl(error_handling=lambda m, e: print(f'Error while loading {m} : {e}'))))
        print(len(df))
        new_df = df[['func_name', 'original_string', 'code', 'code_tokens', 'docstring', 'docstring_tokens']]
        new_df.to_pickle(f'{args.jsonl_output_path}/{file_name}.pkl')


def run(args):
    jsonl2data(args)
    jcoffee_run(args)
    compile_from_txt(args)
    compilation_to_df(args)


def parse_args():
    parser = argparse.ArgumentParser("Generate The Dataset from Jsonl")
    parser.add_argument('--azure_info', type=str, default='original_dataset/test/')
    parser.add_argument('--jsonl_input_path', type=str, default='original_dataset/test/')
    parser.add_argument('--jsonl_output_path', type=str, default='dataset/pkl/')
    parser.add_argument('--class_path', type=str, default='dataset/class/')
    parser.add_argument('--txt_path', type=str, default='dataset/txt/')
    parser.add_argument('--df_path', type=str, default='dataset/df/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
