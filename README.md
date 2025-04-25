# 🌀 Neural Motion Simulator (MoSim)

Official implementation of:

**[Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning](https://oamics.github.io/mosim_page/)**  

🌐 [Project Page](https://oamics.github.io/mosim_page/) | [![arXiv](https://img.shields.io/badge/arXiv-2504.07095-b31b1b.svg)](https://arxiv.org/abs/2504.07095)


---

## 🎯 Visual Comparisons

<p align="center">
  <img src="assets/teaser1.jpg" width="800"/><br/>
  <img src="assets/teaser2.jpg" width="800"/><br/>
  <img src="assets/teaser3.jpg" width="800"/>
</p>

> For more visual results, including video comparisons across agents, please visit our [website](https://oamics.github.io/mosim_page/).

miniforge3/envs/wuji/lib/python3.12/site-packages/dm_control/suite




---

## 🛠️ Setup & Training

Before training or evaluation, please make sure to install MoSim as a developer package and configure the environment properly.

### 🔹 Step 1: Install MoSim in editable mode

Navigate to the root directory of this repository and run:

```bash
pip install -e .
```
Once MoSim is installed, you can start training with the following command:
``` 
python train.py --config config/file/path --device cuda:0
```

export PYTHONPATH="/home/chenjiehao/projects:$PYTHONPATH"
jupyter kernelspec list

{
    "argv": [
        "python",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
        "PYTHONPATH=/path/to/your/modules:$PYTHONPATH"
    ],
    "display_name": "Python 3",
    "language": "python"
}


