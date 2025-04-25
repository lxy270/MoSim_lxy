# 🌀 Neural Motion Simulator (MoSim)

Official implementation of:

**[Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning](https://oamics.github.io/mosim_page/)**  

🌐 [Project Page](https://oamics.github.io/mosim_page/) | 📄 [arXiv](https://arxiv.org/abs/2504.07095)

---

## 🎯 Visual Comparisons

<p align="center">
  <img src="assests/teaser1.jpg" width="800"/><br/>
  <img src="assests/teaser2.jpg" width="800"/><br/>
  <img src="assests/teaser3.jpg" width="800"/>
</p>

> For more visual results, including video comparisons across agents, please visit our [website](https://oamics.github.io/mosim_page/).


---

## 🚀 Highlights

- 🔁 Predicts long-horizon physical dynamics accurately  
- 🧠 Enables sample-efficient skill learning and planning  
- 🎯 Competitive zero-shot RL performance  
- 🧩 Modular — decouples world model and RL algorithm



## Train model
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


python train.py --config /home/chenjiehao/projects/Neural-Simulator/configs/vsdreamer/humanoid/td_mpc.yaml --device cuda:1
