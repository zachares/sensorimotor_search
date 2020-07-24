#!/bin/sh

# first navigate to the folder where you have anaconda installed -- assuming you have installed Anaconda 3
conda create -n sens_search1 python=3.7
conda activate sens_search1

### Setting environmental variables for running mujoco and activating renderer
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

# make sure these paths are correct
echo '#!/bin/sh' >> ./etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scr-ssd/software_packages/mujoco/mujoco200/bin' >> ./etc/conda/activate.d/env_vars.sh
echo 'export MUJOCO_PY_MUJOCO_PATH=/scr-ssd/software_packages/mujoco/mujoco200/' >> ./etc/conda/activate.d/env_vars.sh
echo 'export MUJOCO_PY_MJKEY_PATH=/scr-ssd/software_packages/mujoco/mjkey.txt' >> ./etc/conda/activate.d/env_vars.sh

source ./etc/conda/activate.d/env_vars.sh

echo '#!/bin/sh' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset LD_LIBRARY_PATH' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset MUJOCO_PY_MUJOCO_PATH' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset MUJOCO_PY_MJKEY_PATH' >> ./etc/conda/deactivate.d/env_vars.sh

# make sure this alias is going to the right bash file either bashrc.user or create a bash_aliases
echo "alias onscreen_render_muj='export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so'" >> ~/.bash_aliases
echo "alias offscreen_render_muj='unset LD_PRELOAD'" >> ~/.bash_aliases

# installing pytorch make sure you have the right cudatoolkit for your computer
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

conda install -c conda-forge tensorboardx

conda install -c intel scikit-learn
# echo 'export USE_DAAL4PY_SKLEARN=YES' >> ./etc/conda/activate.d/env_vars.sh
# echo 'unset USE_DAAL4PY_SKLEARN' >> ./etc/conda/deactivate.d/env_vars.sh

conda install -c conda-forge gym

conda install h5py

# create project directory
mkdir /scr-ssd/sens_search_new/
cd /scr-ssd/sens_search_new/

# adding and installing robosuite
git clone -b peter_devel https://github.com/stanford-iprl-lab/robosuite.git

pip install hjson
pip install pyquaternion

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

pip install mujoco-py

pip install pybullet==1.9.5

cd robosuite
pip install -e .
cd ..

pip install gtimer

git clone -b sens_search https://github.com/amichlee/rlkit.git
cd rlkit
pip install -e .
cd ..

git clone https://github.com/vitchyr/viskit.git
cd viskit
pip install -e .
cd ..

git clone -b sens_search  https://github.com/zachares/supervised_learning.git
cd supervised_learning
pip install -e .
cd ..

pip install pyyaml



