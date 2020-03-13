# aa228_finalproject

# Set up your virtual environment

command: pip install virtualenv

command: virtualenv -p /usr/bin/python3.7.4 sens_search

command: source |path to virtual environment|/sens_search/bin/activate

command: pip install -r requirements.txt

# To run an experiment
 
1 activate your new terminal 

command: sens_search

2 set your experiment parameters in framework_parms.yml using a text editor

3 move to folder with code in it 

command: cd |path to code|

4 run code 

command: python framework_evaluation.py

# To look at the results

0 activate your new terminal 

command: sens_search

1 move to results folder 

command: cd |path to results folder|

2 open the results 

command: tensorboard --logdir=. --port <any number between [1000, 9999]>

3 view the results by clicking on the web browser link shown below the command
