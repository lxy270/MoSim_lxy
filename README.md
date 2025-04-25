# Neural Simulator

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
