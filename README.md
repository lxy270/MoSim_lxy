# 🌀 Neural Motion Simulator (MoSim)

Official implementation of:

**[Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning](https://oamics.github.io/mosim_page/)**  

🌐 [Project Page](https://oamics.github.io/mosim_page/) | 📄 [arXiv](https://arxiv.org/abs/2504.07095) | [<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace" height="20"> Checkpoints](https://huggingface.co/wujiss1/MoSim_checkpoints)



---

## 🎯 Visual Comparisons

<p align="center">
  <img src="assets/teaser1.jpg" width="800"/><br/>
  <img src="assets/teaser2.jpg" width="800"/><br/>
  <img src="assets/teaser3.jpg" width="800"/>
</p>

> For more visual results, including video comparisons across agents, please visit our [website](https://oamics.github.io/mosim_page/).






---

## 🛠️ Setup & Training

Before training or evaluation, please make sure to install MoSim as a developer package and configure the environment properly.

### 🔹 Step 1: Install MoSim in editable mode

Navigate to the root directory of this repository and run:

```bash
pip install -e .
```

### 🔹 Step 2: Replace the Corresponding Package Files

We provide a `suite.zip` file under the `assets/` folder containing modified files for `dm_control`. You just need to unzip and replace the existing files in your local environment.

#### ✅ Instructions:

1. Locate and unzip the file (e.g., `assets/suite.zip`).

2. Navigate to the following directory where `dm_control` is installed (based on your environment):

3. Replace the contents of this `suite/` folder with the unzipped files.  

---

> 💡 **Tip:** If you're not sure where `dm_control` is installed, run the following command:

```bash
python -c "import dm_control; print(dm_control.__file__)"
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


