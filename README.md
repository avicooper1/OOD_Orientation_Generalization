For a demo jupyter notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avicooper1/OOD_Orientation_Generalization/blob/master/demo/demo.ipynb)

# Rotation Generalization

This is a collection of tools to run experiments and evaluations for a project studying the ability of DCNN's to generalize to unseen poses of 3D objects.

## Directory Tree
```
Rotation-Generalization/
├── README.md
├── dataset_path.py     #converts dataset attributes to a path to the correct directory. 
│                           offers a class method that converts an index to a dataset path
├── exps.csv            #list of all experiments with experimental variables
├── istarmap.py         #tool used for pooling and tqdm
├── my_dataclasses.py
├── my_models
│   ├── C8SteerableCNN.py   # Equivariant model
│   ├── CORnet_S.py
│   └── CORnet_Z.py
├── notebooks
│   ├── evaluation_vis.ipynb
│   ├── network_analysis.ipynb
│   ├── tools.ipynb
│   └── tools.py
├── render
│   ├── generate_sm_object.py
│   ├── merge_datasets.py
│   ├── model_paths2.json
│   ├── render.py
│   └── render_check.py
├── slurm
│   ├── submit_render.sh
│   └── submit_training.sh
└── train
    ├── dataset.py
    ├── remaining_jobs.json
    ├── run.py
    ├── train.py
    └── training_check.py
```

## Directory Paths

- **Datasets**
    - All datasets can be found at: `/om2/user/avic/datasets`
    - The dataclass available in `dataset_path.py` allows for the specification of a dataset by its parameters. `__repr__()` will return the correct path
        
        ***Example Usage***
        ```python3
        >>> DatasetPath(model_category='plane', type=DatasetType.Bin, scale=True, restriction_axes=(1,2))
        '/home/avic/om2/datasets/plane/bin/mid_scaled/Y_Z'
        ```
     - Alternatively, a class method is provided that allows for the specification of a dataset with its unique index
        
        ***Example Usage***
        ```python3
        >>> DatasetPath.get_dataset(25)
        '/home/avic/om2/datasets/lamp/bin/X_Y'
        ```
- **Experiments**
    - All experiments can be found at `/om2/user/avic/experiments`
    - The dataclass `ExpData` found in `my_dataclasses.py` allows for the specification of an experiment by its parameters. Upon initialization this class will generate all the necessary paths for the experiment directory tree, as exhibited below.
        - To prevent issues with logging, in order to manually print the values of an `ExpData` object, `repr` must be called as `__repr__(print=True)`
        
        ***Example Usage***
        ```python3
        >>> exp_data = ExpData(job_id=4, data_div=20, model_type=ModelType.Inception, pretrained=False, num='23', training_category='plane', testing_category='plane', hole=1, augment=False, scale=True, restriction_axes=(0,1), lr=0.001, batch_size=128, max_epoch=15)
        >>> print(exp_data.__repr__(print=True))
        '''job_id : 4
        data_div : 20
        model_type : ModelType.Inception
        pretrained : False
        num : 23
        training_category : plane
        testing_category : plane
        hole : 1
        augment : False
        scale : True
        restriction_axes : (0, 1)
        lr : 0.001
        batch_size : 128
        max_epoch : 15
        name : Div20
        dir : /home/avic/om2/experiments/exp23
        logs_dir : /home/avic/om2/experiments/exp23/logs
        eval_dir : /home/avic/om2/experiments/exp23/eval
        stats_dir : /home/avic/om2/experiments/exp23/stats
        checkpoints_dir : /home/avic/om2/experiments/exp23/checkpoints
        tensorboard_logs_dir : /home/avic/om2/experiments/exp23/tensorboard_logs/4
        logs : /home/avic/om2/experiments/exp23/logs/Div20.txt
        eval : /home/avic/om2/experiments/exp23/eval/Div20.csv
        testing_frame_path : /home/avic/om2/experiments/exp23/eval/TestingFrame_Div20.csv
        stats : /home/avic/om2/experiments/exp23/stats/Div20.csv
        checkpoint : /home/avic/om2/experiments/exp23/checkpoints/Div20.pt'''
      ```
    - Alternatively, a class method is provided that allows for the specification of an experiment with its unique index
        
        ***Example Usage***
        ```python3
        >>> exp_data = ExpData.get_experiments(4)
        # `exp_data` can be printed out as above 
        ```
    - All experiments live in the following directory structure:
    ```
    /om2/user/avic/experiments/exp*/
    ├── checkpoints     # model checkpoints for further analysis
    │   ├── Div10.pt
    │   ├── Div20.pt
    │   ├── Div30.pt
    │   └── Div40.pt
    ├── eval            # TestingFrames are a dataframe with image atrributes for images in the testing set
    │                   # Div* are dataframes with records of the model's prediction for each image
    │                   # by comparing the predictions in Div* with the expectation in TestingFrames, accuracy can be calculated
    │                   # *_heatmap_id.npy is the in-distribution and *_heatmap_ood.npy is the out of distribution
    │                   # per-orientation accuracy of the experiment. These are generated by running
    │                   # Rotation-Generalization/analysis/generate_eval_heatmaps.py
    │   ├── Div10.csv
    │   ├── Div10_heatmap_id.npy
    │   ├── Div10_heatmap_ood.npy
    │   ├── Div20.csv
    │   ├── Div20_heatmap_id.npy
    │   ├── Div20_heatmap_ood.npy
    │   ├── Div30.csv
    │   ├── Div30_heatmap_id.npy
    │   ├── Div30_heatmap_ood.npy
    │   ├── Div40.csv
    │   ├── Div40_heatmap_id.npy
    │   ├── Div40_heatmap_ood.npy
    │   ├── TestingFrame_Div10.csv
    │   ├── TestingFrame_Div20.csv
    │   ├── TestingFrame_Div30.csv
    │   └── TestingFrame_Div40.csv
    ├── logs           # Text files with properties and logs of each experiment
    │   ├── Div10.txt
    │   ├── Div20.txt
    │   ├── Div30.txt
    │   └── Div40.txt
    ├── stats          # The per-epoch statistics (accuracies, loss) of each experiment
    │   ├── Div10.csv
    │   ├── Div20.csv
    │   ├── Div30.csv
    │   └── Div40.csv
    └── tensorboard_logs    # Directory of tensorboard logs for use with tensorboard
        └── ...

    ```

## Usage

#### Rendering
- Synthetic Datasets
    - For easy generation of all datasets, `dataset_paths.py` has a class method that converts an index into the path for a certain dataset
        - Each synthetic dataset has 500 annotation files (explained below) and with a total of 4 synthetic datasets, values ranging from `[1-1999]` are valid indices
    - Each dataset lives at a path generated by `dataset_paths.py` In its directory there is a directory for all the images, an annotation file for each category, a merged annotation file for all the categories, and optionally further nested directories
        - An annotation file contains data for each image including the path to the image, the object's category, the object's rotation etc.
    - **Rendering Pipeline**
        - (optionally:) Use the tool `python render/render_check.py` to get a bash-formatted list of rendering jobs that aren't yet completed
        - set the array variable in `sbatch slurm/submit_render.sh` with the indices of the datasets to be generated, either with a range or with a list
            - this list can be obtained with the above mentioned `render_check`
        - obtain the merged dataset annotation files by running `python render/merge_datasets.py`
        
        ***Example Usage***
        ```shell script
        python render/render_check.py
            # use the output as the array in submit_render.sh
        sbatch slurm/submit_render.sh
        python render/merge_datasets.py
        ```
        
 
 - iLab Datasets
    - Updated code for iLab has not yet been implemented. This is forthcoming

#### Training
- For easy enumeration of experiments, `ExpData` in `my_dataclasses.py` provides a constructor that takes experimental parameters as well as a class method to obtain a specific experiment from an index
- All experiments can be found in `exps.csv` This contains all the experiments to be run, that is all the `ExpData`'s with all legal indices
    - The `job_id` is the (render) indices equivalent for training
    - Note: `exps.csv` is not continuous in that it does not include experiments that use augmented data. This is due to a new pipeline that compares unaugmented experiments with a analytically generated 2D heatmap 
- Experiments live in directories with an `exp_num` as listed in their `ExpData`
- Currently experiments are run for 10 epochs. This is a hyperparameter and we might consider changing it for all networks or at least for some networks
- **Training Pipeline**
    - Use the tool `python train/training_check.py` to generate a file at `train/reamining_jobs.json`with a list of experiments that aren't yet completed
    - set the array variable in `sbatch slurm/submit_training.sh` with the range printed out by the previous tool
        - there are 2 options for gpu specification. Resnet-18 models can be run on `tesla-k80` gpu's, so when training these models uncomment the relevant line in the sbatch file. When training larger models, uncomment the lines that request `high-capacity` gpu's with 11GB of memory
    
    ***Example Usage***
    ```shell script
    python train/training_check.py
    sbatch slurm/submit_training.sh
    ```
  
## Full Experiment Proposal
For complete analysis, we plan to run the following experiments. We will 

Datasets
Synthetic (for each of the following groups):
Categories:
Plane
Car	
Lamp
Shepard Metzler
Full
Scaled
Unscaled
Bin (Generate Mid Scale, split bins in hole and center)
X, Y
Y, Z
X, Z
Total: 4 · ((2 + 3) · 50) = 1,000
iLab
Backbones
Resnet-18
DenseNet
InceptionV3
https://pytorch.org/docs/stable/torchvision/models.html
Equivariant Group CNN’s
(Possible mention of CorNet, for recurrence)
Data Augmentation
For translation - possibly only do in testing, not even in training
Small translations
Transfer from one object to another
Transfer from ImageNet
One of two backbones (Resnet, CorNet)

        