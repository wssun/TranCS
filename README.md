# TranCS: Code search based on Context-aware Code Translation
## Environment
ubuntu 16.04
### Requirements
python==3.8  
torch==1.8.1  
tables==3.6.1  
h5py==3.3.0  
tqdm==4.62.3  

## Dataset
### Original dataset
The original dataset can be downloaded from [https://github.com/github/CodeSearchNet](https://github.com/github/CodeSearchNet)

### Process dataset
1. put the original java dataset in **process_dataset/original_dataset**
2. run the processing code
```shell
python process_dataset/generate_from_jsonl.py
```
3. convert instruction to translation 
```shell
python process_instruction/instruction2tran.py
```
4. run the preprocessing code
```shell
python src/dataset_utils.py
```

## Baseline
DeepCS: [https://github.com/guxd/deep-code-search](https://github.com/guxd/deep-code-search)  
MMAN: [https://github.com/wanyao1992/mman_public](https://github.com/wanyao1992/mman_public)
### Train
```shell
python baseline_methods/DeepCS/train.py
python baseline_methods/MMAN/train.py
```
### Test
```shell
python baseline_methods/DeepCS/test.py
python baseline_methods/MMAN/test.py
```

## TranCS
We provide TranCS model and raw data processing code which are listed in src/ and process_instruction/ directories.
### Train the Model
```shell
python src/train.py
```
### Test the Model
```shell
python src/test.py
```