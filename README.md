# TranCS: Code search based on Context-aware Code Translation
## TranCS
We provide TranCS model and raw data processing code which are listed in src/ and process_instruction/ directories.

## Dataset
The original dataset can be downloaded from [https://github.com/github/CodeSearchNet](https://github.com/github/CodeSearchNet)

## Running the Model
### Convert Instruction to Translation 
```shell
python process_instruction/instruction2tran.py
```
### Train the Model
```shell
python src/train.py
```
### Test the Model
```shell
python src/test.py
```