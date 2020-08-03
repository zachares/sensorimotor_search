import sys
import yaml
from dataloader import Custom_DataLoader
import perception_models as pm
from train_using_supervised_learning import *
import utils_sl as sl
import torch
#########################################
#### CODE DOES NEED TO BE MODIFIED ######
#########################################
def project_main():
	with open("learning_params.yml", 'r') as ymlfile:
	    cfg = yaml.safe_load(ymlfile)

	use_cuda = cfg['training_params']['use_GPU'] and torch.cuda.is_available()

	device = torch.device("cuda" if use_cuda else "cpu")

	data_loader, val_data_loader = sl.init_dataloader(cfg, Custom_DataLoader, device)
	ref_model_dict = pm.get_ref_model_dict()
	model_dict = sl.declare_models(ref_model_dict, cfg, device)

	train_using_supervised_learning(cfg, model_dict, data_loader, val_data_loader, device)

if __name__ == "__main__":
	project_main()