# Missing You-ML
### 💡2024 GDSC Solution Challenge
Team Iris's ML Repository

## Installation
### environment
> Ensure that Python 3.9+ is installed on your system. You can check this by using: `python --version`.  

```bash
conda create -n MissingYouML python=3.9
conda activate MissingYouML
```
### dependencies
```bash
pip install -r requirements.txt
```

## Run
```bash
If you've trained your own arcface model:
    uvicorn main:app
Otherwise:
    uvicorn main2:app
