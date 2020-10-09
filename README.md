# REPORT

This is the **REPORT** code for fake news detection. We are currently using **twitter15&16 dataset**

### Current Structure:
Project structures of this repo are listed in project_structure.txt

### requirements:
1.[torch_geometric](https://github.com/rusty1s/pytorch_geometric):
(also called 'PyG')

torch_geometric does not support conda installation currently, please view
your installing versions at [here](https://pytorch-geometric.com/whl/torch-1.5.0.html), 
and install it with **pip**.
you need to install **4** dependencies before installing PyG: 

+ torch-scatter
+ torch-sparse
+ torch-cluster
+ torch-spline-conv
<br>
<br>

##### an example of installation:

+ Python: 3.8.5
+ Pytorch: 1.5.0  (torch_geometric is currently available to up to torch 1.5.0)
+ cuda: 10.2

script:

        pip install torch-scatter==2.0.4+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
        pip install torch-sparse==0.6.3+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
        pip install torch-cluster==1.5.4+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
        pip install torch-spline-conv==1.2.0+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
        pip install torch-geometric
        # change 'cu102' to 'cpu' or 'cu101' or others depending on your PyTorch installation. 

###### dataset
all data are put in **datasets/**,
complete dataset is available at: https://www.dropbox.com/home/FakeNew2020/datasets

##### 



#### Steps for pre_processing:

1. in each folder that contain *task.py*, configure the parameter according to your experiments. <br>
2. in *datasets/twitter15(6)/raw_data/*, run *raw_data.bat* <br>
3. in *pre_process/*,  run *pre_process.bat* <br>
4. in *load_data15(6)/*, run *add_label.py* and *count_tweet.py*



