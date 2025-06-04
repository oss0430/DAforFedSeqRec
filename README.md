This research was developed using

<h1 align="center">
    <img src="https://img.alicdn.com/imgextra/i4/O1CN01yp6zdb23HOJJkCmZg_!!6000000007230-2-tps-2048-1009.png" width="400" alt="federatedscope-logo">
</h1>

![](https://img.shields.io/badge/language-python-blue.svg)
![](https://img.shields.io/badge/license-Apache-000000.svg)
[![Website](https://img.shields.io/badge/website-FederatedScope-0000FF)](https://federatedscope.io/)
[![Playground](https://shields.io/badge/JupyterLab-Enjoy%20Your%20FL%20Journey!-F37626?logo=jupyter)](https://try.federatedscope.io/)
[![Contributing](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://federatedscope.io/docs/contributor/)

FederatedScope is a comprehensive federated learning platform that provides convenient usage and flexible customization for various federated learning tasks in both academia and industry.  Based on an event-driven architecture, FederatedScope integrates rich collections of functionalities to satisfy the burgeoning demands from federated learning, and aims to build up an easy-to-use platform for promoting learning safely and effectively.

A detailed tutorial is provided on this website: [federatedscope.io](https://federatedscope.io/)

You can try FederatedScope via [FederatedScope Playground](https://try.federatedscope.io/) or [Google Colab](https://colab.research.google.com/github/alibaba/FederatedScope).

## Quick Start

### Step 1. Installation

First of all, users need to clone the source code and install the required packages (we suggest python version >= 3.9). You can choose between the following two installation methods (via docker or conda) to install FederatedScope.

```bash
git clone https://github.com/alibaba/FederatedScope.git
cd FederatedScope
```
#### Use Docker

You can build docker image and run with docker env (cuda 11 and torch 1.10):

```
docker build -f environment/docker_files/federatedscope-torch1.10.Dockerfile -t alibaba/federatedscope:base-env-torch1.10 .
docker run --gpus device=all --rm -it --name "fedscope" -w $(pwd) alibaba/federatedscope:base-env-torch1.10 /bin/bash
```
If you need to run with down-stream tasks such as graph FL, change the requirement/docker file name into another one when executing the above commands:
```
# environment/requirements-torch1.10.txt -> 
environment/requirements-torch1.10-application.txt

# environment/docker_files/federatedscope-torch1.10.Dockerfile ->
environment/docker_files/federatedscope-torch1.10-application.Dockerfile
```
Note: You can choose to use cuda 10 and torch 1.8 via changing `torch1.10` to `torch1.8`.
The docker images are based on the nvidia-docker. Please pre-install the NVIDIA drivers and `nvidia-docker2` in the host machine. See more details [here](https://github.com/alibaba/FederatedScope/tree/master/environment/docker_files).

#### Use Conda

We recommend using a new virtual environment to install FederatedScope:

```bash
conda create -n fs python=3.9
conda activate fs
```

If your backend is torch, please install torch in advance ([torch-get-started](https://pytorch.org/get-started/locally/)). For example, if your cuda version is 11.3 please execute the following command:

```bash
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 torchtext=0.11.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

For users with Apple M1 chips:
```bash
conda install pytorch torchvision torchaudio -c pytorch
# Downgrade torchvision to avoid segmentation fault
python -m pip install torchvision==0.11.3
```

Finally, after the backend is installed, you can install FederatedScope from `source`:

##### From source

```bash
# Editable mode
pip install -e .

# Or (developers for dev mode)
pip install -e .[dev]
pre-commit install
```

Now, you have successfully installed the minimal version of FederatedScope. (**Optinal**) For application version including graph, nlp and speech, run:

```bash
bash environment/extra_dependencies_torch1.10-application.sh
```

### Step 2. Prepare datasets for Federated Sequential Recommendation System

To run the Federated Sequential Recommendation runs we must prepare user-item interaction dataset. We recommend using dataset from Recbole (https://recbole.io/dataset_list.html).

```bash
python augmentation/transform_str_id_to_int_id.py --input_csv your_csv_path.csv --output_dir your_dir
```
This act will create 3 files that specifies the user-item interaction
1. your_dir/inter.csv the main file that specify the interaction
2. your_dir/item.csv the file specifies the map of item name and its ids
3. your_dir/user.csv the file specifies the map of user name and its ids

If you wish to use your own custom Dataset, specify the user, item, timestamp columns using
```bash
python augmentation/transform_str_id_to_int_id.py -i your_item_column  -u your_user_column -t your_timestamp_column --input_csv your_csv_path.csv --output_dir your_dir 
```

### Step 3. Prepare models (Optional)

Currently SASRec is the only available model. But Users can set up `cfg.model.type = MODEL_NAME` to apply a specific model architecture in FL tasks. For example,

```yaml
cfg.model.type = 'BERT4Rec'
```
However this requires you to setup a source code at  ```federatedscope/contrib/model```
Setup for SASRec can be seen in ```federatedscope/contrib/model/sasrec.py```
Currently Sequential Recommendation Runs allows only interaction sequence inputs. To modify this I/O we recommend looking at ```federatedscope/contrib/data/sr_data.py```, ```federatedscope/contrib/trainer/sasrec_trainer.py``` and ```federatedscope/contrib/metrics/recall_ndcg.py```.

FederatedScope allows users to use customized models via registering. Please refer to [Customized Models](https://federatedscope.io/docs/own-case/#model) for more details about how to customize a model architecture.

### Step 4. Start running an Federated Sequential Recommendation

To run the Federated Sequential Recommendation (FSR) runs we need a configuration file and the dataset. 
The path of the dataset must be specified at your configuration file.
The example of this configuration file can be seen in
```run_configs/base_experiments/amazon_sports/Amazon_Sports_NoAugs.yaml```

The core configurations are :
```yaml
federate.clinet_num : Number of users in the total dataset

data.root : Path to dir of "your_inter.csv"
data.partitioned_df_path : Path to where we save the train/test/valid version of your_inter.csv 
data.user_num : Number of users in the total dataset
data.item_num : Number of items in the total dataset
data.min_sequence_length : A minimum length of the sequence 
data.max_sequence_length : A maximum length of the sequence

model.item_num : Number of items in the total dataset

```

Also Keep this configurations to :
```yaml
data.consistent_label_distribution : False
data.type : sr_data
```

If the configuration is complete simply run.
```bash
python main.py --cfg run_configs/base_experiments/amazon_sports/Amazon_Sports_NoAugs.yaml
```

NOTE that this act will create the splited version of inter.csv at the ```data.partitioned_df_path```. 3 train/valid/test csv files can be reused for faster initiations.



## Using Augmentation
This act requires you to run "Quick Start - Step 4" once. Since it requires the Train/Valid/Test Split files.

The location of the splited data files will be located with specified path at the correspondinf yaml file
```yaml
data.partitioned_df_path: your_splited_files_dir
```


### Step 1. Configure for the original dataset
This is an optional step if you wish to run with Augmented FL.
To run this, we first need to configurate paths at ```augmentation/seq_rec_augmentation.py```.
Example of this configurations are given at line 46 to 56 of the corresponding source file.

```
ml_1m_configs = {
    "train_dataframe_path" : '../../../../data1/donghoon/FederatedScopeData/ml-1m/split/train.csv',
    "result_branch_path" : '../../../../data1/donghoon/FederatedScopeData/ml-1m/',
    "user_column" : 'user_id:token',
    "item_column" : 'item_id:token',
    "timestamp_column" : 'timestamp:float',
    "max_item_ids" : 3952,
    "max_sequence_length" : 200,
    "min_sequence_length" : 3
}
```
The specified configuration must be mentioned at ```augmentation/seq_rec_augmentation.py def load_dataset_configs``` (line 655)


### Step 2. Generate Augmentation
6 types of augmentation methods are availiable. ```'random_replacing' 'cutting', 'shuffle', 'random_pushing', 'self_sampled_pushing', 'random_deletion'```

All the minor augmentation details can be found at ```augmentation/seq_rec_augmentation.py``` line(22~42)

Simply run
```bash
python augmentation/seq_rec_augmentation.py -o your_configured_original_dataset_name -n number_of_generated_augmented_seuqences -t augmentation_type -no_org True
```

For simple explanation of the argumentation arguments. 
```
"args" : [
                "-o", "amazon_beauty", // original dataset name
                "-n", "60",
                "-r", "", //"my_augmented_trainset.csv", // result dataset name (if not given, it will be original dataset name + augmentation type)
                "-t", "random_masking", // augmenation type
                "-d", "left", // direction, only used at cutting augmentation
                "-p", "0.1", // replace probabiliy, only used at replace augmentation
                "-ls", "2", // push length range start, only used at pushing augmentation
                "-le", "3", // push length range end, only used at pushing augmentation
                "-is", "1", // item_perturb range start
                "-ie", "5", // item_perturb range end
                "-mi", "0", // mask count
                "-no_org", "True"
            ]
```

### Step 3. Start running an Augmented FSR

Modify the configurations for Augmented Training sets. Simply add the following augmentation configurations at ```data.augmentation_args```
```yaml
data :
  augmentation_args :
    augmentation_column : 'augmentation_idx:token'
    use_augmentation : True
    max_augmentation_idx : 10 ## load 10 augmented sequence for each client
    is_zero_original : False
    is_multiple : True
    aug_types_count : 2 ## aug types
    df_paths : ['original/split',
                'my_dir_to_augmented_train_set'] ## this length should be aug_types_count
```

The example can be seen at ``` run_configs/base_experiments/amazon_sports/Amazon_Sports_RandomDeletion.yaml ```

After modification you can start by
```bash
python main.py --cfg run_configs/base_experiments/amazon_sports/Amazon_Sports_RandomDeletion.yaml
```