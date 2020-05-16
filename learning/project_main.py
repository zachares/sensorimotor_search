import sys
import yaml

#########################################
#### CODE DOES NEED TO BE MODIFIED ######
#########################################

with open("learning_params.yml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

relative_path = cfg["relative_path"]

sys.path.insert(0, relative_path)

from main import *

def project_main():
    main()

if __name__ == "__main__":
	project_main()