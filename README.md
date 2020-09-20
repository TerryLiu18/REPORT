# HierarchyGCN

This is the HierarchyGCN code for fake news detection. We are currently using **twitter16 dataset**

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

###### usage
1. to make all route available in the repo, please make sure opening this project as name: 'MyGAT',

        project_name = 'MyGAT'
        loc = os.path.abspath(os.path.dirname(__file__)).split(project_name)[0]
        root_path = pth.join(loc, project_name) 

2. run **utils.py** to get some auxiliary files

3. run **user_profile_processing.py** and **tweet_profile_processing.py** to get processed files and word sets. 


###### dataset
all data are put in **datasets/**,
complete dataset is available at: https://www.dropbox.com/home/FakeNew2020/datasets

##### 

#### Steps for pre_processing:
not complete yet

0. all_user_stat.py -> all_user_info.csv<br>
1. python utils.py<br>
2. python user_processing.py<br>
3. python tweet_processing.py<br>
4. python tree_processing.py<br>
5. python build_graph_connection.py<br>
6. python build_tree_connection.py<br>
7. python word_mapping.py<br>

check_connection<br>


8. add add_label.py in load_data{}<br>