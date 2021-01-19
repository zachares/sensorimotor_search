# Sensorimotor Search:

This is the code associated with the research project "Interpreting Contact Interactions to Overcome Failure in Robotic Assembly Tasks". A copy of the work can be found on arxiv at the link: https://arxiv.org/abs/2101.02725

# Installation Instructions

#!/bin/sh

#Assuming you have installed Anaconda 3
conda create -n sens_search python=3.7
conda activate sens_search

#Setting environmental variables for running mujoco and activating renderer
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

#CHECK PATHS TO MUJOCO make sure these paths are correct
echo '#!/bin/sh' >> ./etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scr-ssd/software_packages/mujoco/mujoco200/bin' >> ./etc/conda/activate.d/env_vars.sh
echo 'export MUJOCO_PY_MUJOCO_PATH=/scr-ssd/software_packages/mujoco/mujoco200/' >> ./etc/conda/activate.d/env_vars.sh
echo 'export MUJOCO_PY_MJKEY_PATH=/scr-ssd/software_packages/mujoco/mjkey.txt' >> ./etc/conda/activate.d/env_vars.sh

source ./etc/conda/activate.d/env_vars.sh

echo '#!/bin/sh' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset LD_LIBRARY_PATH' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset MUJOCO_PY_MUJOCO_PATH' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset MUJOCO_PY_MJKEY_PATH' >> ./etc/conda/deactivate.d/env_vars.sh

#CHECK TO MAKE SURE ITS THE RIGHT BASH FILE make sure this alias is going to the right bash file either bashrc.user or create a bash_aliases
echo "alias onscreen_render_muj='export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so'" >> ~/.bash_aliases
echo "alias offscreen_render_muj='unset LD_PRELOAD'" >> ~/.bash_aliases

#Installing pytorch make sure you have the right cudatoolkit for your computer
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

conda install -c conda-forge tensorboard
conda install -c conda-forge tensorboardx

conda install -c intel scikit-learn
#echo 'export USE_DAAL4PY_SKLEARN=YES' >> ./etc/conda/activate.d/env_vars.sh
#echo 'unset USE_DAAL4PY_SKLEARN' >> ./etc/conda/deactivate.d/env_vars.sh

conda install -c conda-forge gym

conda install h5py
conda install matplotlib

#CHECK PATH create project directory
mkdir /scr-ssd/sens_search/
cd /scr-ssd/sens_search/

#Adding and installing robosuite
git clone -b peter_devel https://github.com/stanford-iprl-lab/robosuite.git

pip install hjson
pip install pyquaternion
pip install pyyaml
pip install mujoco-py
pip install pybullet==1.9.5
pip install gtimer

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

cd robosuite
pip install -e .
cd ..

#git clone -b https://github.com/zachares/rlkit.git
#cd rlkit
#pip install -e .
#cd ..

#git clone https://github.com/vitchyr/viskit.git
#cd viskit
#pip install -e .
#cd ..

git clone -b https://github.com/zachares/supervised_learning.git
cd supervised_learning
pip install -e .
cd ..

git clone -b sens_search https://github.com/zachares/sensorimotor_search.git
cd sensorimotor_search
pip install -e .
cd ..





