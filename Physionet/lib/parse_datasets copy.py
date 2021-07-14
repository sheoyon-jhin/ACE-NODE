
import os
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from lib.diffeq_solver import DiffeqSolver
from generate_timeseries import Periodic_1d
from torch.distributions import uniform

from torch.utils.data import DataLoader
from mujoco_physics import HopperPhysics
from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
from person_activity import PersonActivity, variable_time_collate_fn_activity

from sklearn import model_selection
import random

import pandas as pd 
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
import os 
ROOT_PATH = os.path.dirname(os.path.abspath(__file__+"/../"))
#####################################################################################################
def parse_datasets(args, device):
	
	
	def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
		batch = torch.stack(batch)
		data_dict = {
			"data": batch, 
			"time_steps": time_steps}

		data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
		return data_dict


	dataset_name = args.dataset

	n_total_tp = args.timepoints + args.extrap
	max_t_extrap = args.max_t / args.timepoints * n_total_tp

	##################################################################
	# MuJoCo dataset
	if dataset_name == "hopper":
		dataset_obj = HopperPhysics(root='data', download=True, generate=False, device = device)
		dataset = dataset_obj.get_dataset()[:args.n]
		dataset = dataset.to(device)


		n_tp_data = dataset[:].shape[1]

		# Time steps that are used later on for exrapolation
		time_steps = torch.arange(start=0, end = n_tp_data, step=1).float().to(device)
		time_steps = time_steps / len(time_steps)

		dataset = dataset.to(device)
		time_steps = time_steps.to(device)

		if not args.extrap:
			# Creating dataset for interpolation
			# sample time points from different parts of the timeline, 
			# so that the model learns from different parts of hopper trajectory
			n_traj = len(dataset)
			n_tp_data = dataset.shape[1]
			n_reduced_tp = args.timepoints

			# sample time points from different parts of the timeline, 
			# so that the model learns from different parts of hopper trajectory
			start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp +1, size=n_traj)
			end_ind = start_ind + n_reduced_tp
			sliced = []
			for i in range(n_traj):
				  sliced.append(dataset[i, start_ind[i] : end_ind[i], :])
			dataset = torch.stack(sliced).to(device)
			time_steps = time_steps[:n_reduced_tp]

		# Split into train and test by the time sequences
		train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

		n_samples = len(dataset)
		input_dim = dataset.size(-1)

		batch_size = min(args.batch_size, args.n)
		train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
			collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "train"))
		test_dataloader = DataLoader(test_y, batch_size = n_samples, shuffle=False,
			collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "test"))
		
		data_objects = {"dataset_obj": dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader)}
		return data_objects

	##################################################################
	# Physionet dataset

	if dataset_name == "physionet":
		
		train_dataset_obj = PhysioNet('data/physionet', train=True, 
										quantization = args.quantization,
										download=True, n_samples = min(10000, args.n), 
										device = device)
		# Use custom collate_fn to combine samples with arbitrary time observations.
		# Returns the dataset along with mask and time steps
		test_dataset_obj = PhysioNet('data/physionet', train=False, 
										quantization = args.quantization,
										download=True, n_samples = min(10000, args.n), 
										device = device)

		# Combine and shuffle samples from physionet Train and physionet Test
		total_dataset = train_dataset_obj[:len(train_dataset_obj)]
		
		if not args.classif:
			# Concatenate samples from original Train and Test sets
			# Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
			total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]
		# import pdb ; pdb.set_trace()
		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, 
			random_state = 42, shuffle = True)
		
		record_id, tt, vals, mask, labels = train_data[0]
		
		
				
		# import pdb ; pdb.set_trace()
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 

		batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
		data_min, data_max = get_data_min_max(total_dataset)

		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max))
		test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))

		attr_names = train_dataset_obj.params
		data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"attr": attr_names, #optional
					"classif_per_tp": False, #optional
					"n_labels": 1} #optional
		return data_objects

	##################################################################
	# CharacterTrajectory dataset
	if dataset_name =="CharacterTrajectory":
		train_X= torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/0UEA_base/train_X.pt')
		val_X = torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/0UEA_base/val_X.pt')
		test_X=torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/0UEA_base/test_X.pt')
		train_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/0UEA_base/train_y.pt')
		val_y =torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/0UEA_base/val_y.pt')
		test_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/0UEA_base/test_y.pt')
		train_data = [] 
		val_data=[]
		test_data=[]
		
		for i in range(len(train_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			vals = train_X[i,:,:].cuda()
			mask = torch.ones_like(train_X[i,:,:]).cuda()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,train_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			train_data.append(temp)
		print("finish >> train data set")
		for i in range(len(val_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			vals = val_X[i,:,:].cuda()
			mask = torch.ones_like(val_X[i,:,:]).cuda()
			labels = torch.zeros([182,20]).cuda()
			labels[:,val_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			val_data.append(temp)
		print("finish >> validation data set")
		for i in range(len(test_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			vals = test_X[i,:,:].cuda()
			mask = torch.ones_like(test_X[i,:,:]).cuda()
			labels = torch.zeros([182,20]).cuda()
			labels[:,test_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			test_data.append(temp)
		print("finish >> test data set")		
		
		total_dataset = train_data+val_data+test_data 
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 
		batch_size = min(min(len(train_data), args.batch_size), args.n)
		

		input_dim = vals.size(-1)

		

		
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		val_dataloader = DataLoader(val_data, batch_size= len(val_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))
		
		test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches" : len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": True, #optional
					"n_labels": labels.size(-1)}
		return data_objects

	if dataset_name =="CharacterTrajectory30":
		train_X= torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/30UEA_base/train_X.pt')
		val_X = torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/30UEA_base/val_X.pt')
		test_X=torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/30UEA_base/test_X.pt')
		train_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/30UEA_base/train_y.pt')
		val_y =torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/30UEA_base/val_y.pt')
		test_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/30UEA_base/test_y.pt')
		# import pdb;pdb.set_trace()
		# if torch.isnan(
		
		train_data = [] 
		val_data=[]
		test_data=[]
		
		for i in range(len(train_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			v = train_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			# import pdb ; pdb.set_trace()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,train_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			train_data.append(temp)
		print("finish >> train data set")
		for i in range(len(val_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			v = val_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			# import pdb ; pdb.set_trace()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,val_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			val_data.append(temp)
		print("finish >> validation data set")
		for i in range(len(test_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			v = test_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			# import pdb ; pdb.set_trace()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,train_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			test_data.append(temp)
		print("finish >> test data set")		
		
		total_dataset = train_data+val_data+test_data 
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 
		batch_size = min(min(len(train_data), args.batch_size), args.n)
		

		input_dim = vals.size(-1)

		

		
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		val_dataloader = DataLoader(val_data, batch_size= len(val_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))
		
		test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches" : len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": True, #optional
					"n_labels": labels.size(-1)}
		return data_objects


	if dataset_name =="CharacterTrajectory50":
		train_X= torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/50UEA_base/train_X.pt')
		val_X = torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/50UEA_base/val_X.pt')
		test_X=torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/50UEA_base/test_X.pt')
		train_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/50UEA_base/train_y.pt')
		val_y =torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/50UEA_base/val_y.pt')
		test_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/50UEA_base/test_y.pt')
		# import pdb;pdb.set_trace()
		# if torch.isnan(
		
		train_data = [] 
		val_data=[]
		test_data=[]
		
		for i in range(len(train_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			v = train_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			# import pdb ; pdb.set_trace()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,train_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			train_data.append(temp)
		print("finish >> train data set")
		for i in range(len(val_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			v = val_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			# import pdb ; pdb.set_trace()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,val_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			val_data.append(temp)
		print("finish >> validation data set")
		for i in range(len(test_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			v = test_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			# import pdb ; pdb.set_trace()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,train_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			test_data.append(temp)
		print("finish >> test data set")		
		
		total_dataset = train_data+val_data+test_data 
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 
		batch_size = min(min(len(train_data), args.batch_size), args.n)
		

		input_dim = vals.size(-1)

		

		
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		val_dataloader = DataLoader(val_data, batch_size= len(val_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))
		
		test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches" : len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": True, #optional
					"n_labels": labels.size(-1)}
		return data_objects


	if dataset_name =="CharacterTrajectory70":
		train_X= torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/70UEA_base/train_X.pt')
		val_X = torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/70UEA_base/val_X.pt')
		test_X=torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/70UEA_base/test_X.pt')
		train_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/70UEA_base/train_y.pt')
		val_y =torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/70UEA_base/val_y.pt')
		test_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/70UEA_base/test_y.pt')
		# import pdb;pdb.set_trace()
		# if torch.isnan(
		
		train_data = [] 
		val_data=[]
		test_data=[]
		
		for i in range(len(train_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			v = train_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			# import pdb ; pdb.set_trace()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,train_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			train_data.append(temp)
		print("finish >> train data set")
		for i in range(len(val_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			v = val_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			# import pdb ; pdb.set_trace()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,val_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			val_data.append(temp)
		print("finish >> validation data set")
		for i in range(len(test_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(182)).cuda()
			v = test_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			# import pdb ; pdb.set_trace()
			
			labels = torch.zeros([182,20]).cuda()
			
			labels[:,train_y[i]]  = 1
			temp = (record_id,tt,vals,mask,labels)
			test_data.append(temp)
		print("finish >> test data set")		
		
		total_dataset = train_data+val_data+test_data 
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 
		batch_size = min(min(len(train_data), args.batch_size), args.n)
		

		input_dim = vals.size(-1)

		

		
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		val_dataloader = DataLoader(val_data, batch_size= len(val_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))
		
		test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches" : len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": True, #optional
					"n_labels": labels.size(-1)}
		return data_objects

	##################################################################
	# Stock dataset
	if dataset_name =="stock":
		def make_batch(input_data, seq_len):
			data_x = []
			data_y = []
			data_len = len(input_data)

			for i in range(data_len-seq_len):
				data_seq = input_data[i : i+seq_len]
				data_label = input_data[i+seq_len : i+seq_len+1]
				data_x.append(data_seq)
				data_y.append(data_label)
			return data_x, data_y

		data_path = '/home/bigdyl/socar/NeuralCDE/experiments/datasets/stock_data.csv'
		df = pd.read_csv(data_path)
		data = df.values.astype(float)

		scaler = MinMaxScaler()
		data_norm = scaler.fit_transform(data)

		batch_size = 200
		seq_len = 24
		data_x, data_y = make_batch(data_norm, seq_len)
		train_size = int(len(data_norm) * 0.7)
		val_size = (len(data_norm) - train_size) // 2
		train_X, train_y = data_x[:train_size], data_y[:train_size]
		val_X,val_y = data_x[train_size:train_size+val_size], data_y[train_size:train_size+val_size]
		test_X, test_y = data_x[train_size+val_size:], data_y[train_size+val_size:]

		train_X, train_y = torch.Tensor(train_X), torch.Tensor(train_y)
		val_X, val_y = torch.Tensor(val_X), torch.Tensor(val_y)
		test_X, test_y = torch.Tensor(test_X), torch.Tensor(test_y)

		train_data = [] 
		val_data=[]
		test_data=[]
		
		for i in range(len(train_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			vals = train_X[i,:,:].cuda()
			mask = torch.ones_like(train_X[i,:,:]).cuda()
			labels = train_y[i,:,:].cuda()
			temp = (record_id,tt,vals,mask,labels)
			train_data.append(temp)
		print("finish >> train data set")

		for i in range(len(val_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			vals = val_X[i,:,:].cuda()
			mask = torch.ones_like(val_X[i,:,:]).cuda()
			labels = val_y[i,:,:].cuda()
			temp = (record_id,tt,vals,mask,labels)
			val_data.append(temp)
		print("finish >> validation data set")
		
		for i in range(len(test_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			vals = test_X[i,:,:].cuda()
			mask = torch.ones_like(test_X[i,:,:]).cuda()
			labels = torch.zeros([182,20]).cuda()
			labels = test_y[i,:,:].cuda()
			test_data.append(temp)
		print("finish >> test data set")		
		# import pdb ; pdb.set_trace()

		total_dataset = train_data+val_data+test_data 
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 
		batch_size = min(min(len(train_data), args.batch_size), args.n)
		
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		val_dataloader = DataLoader(val_data, batch_size= len(val_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))
		
		test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches" : len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": False, #optional
					"n_labels": 1}
		return data_objects


	if dataset_name =="stock30":
		train_X= torch.load(ROOT_PATH+'/0Stock_raw/30/train_X.pt')
		val_X = torch.load(ROOT_PATH+'/0Stock_raw/30/val_X.pt')
		test_X=torch.load(ROOT_PATH+'/0Stock_raw/30/test_X.pt')
		train_y= torch.load(ROOT_PATH+'/0Stock_raw/30/train_y.pt')
		val_y =torch.load(ROOT_PATH+'/0Stock_raw/30/val_y.pt')
		test_y= torch.load(ROOT_PATH+'/0Stock_raw/30/test_y.pt')
		
		seq_len = 24
		
		train_data = [] 
		val_data=[]
		test_data=[]
		
		for i in range(len(train_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			v = train_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()
			# import pdb ; pdb.set_trace()
			mask = check + zero.cuda()
			# mask = mask.cuda()
			labels = train_y[i,:,:].cuda()
			temp = (record_id,tt,vals.float(),mask,labels.float())
			train_data.append(temp)
		print("finish >> train data set")

		for i in range(len(val_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			v = val_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()
			mask = check + zero.cuda()
			
			labels = val_y[i,:,:].cuda()
			temp = (record_id,tt,vals.float(),mask,labels.float())
			val_data.append(temp)
		print("finish >> val data set")


		for i in range(len(test_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			v = test_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()
			mask = check + zero.cuda()
			
			labels = test_y[i,:,:].cuda()
			temp = (record_id,tt,vals.float(),mask,labels.float())
			test_data.append(temp)
		print("finish >> test data set")

		# import pdb ; pdb.set_trace()
		total_dataset = train_data+val_data+test_data 
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 
		batch_size = min(min(len(train_data), args.batch_size), args.n)
		
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		val_dataloader = DataLoader(val_data, batch_size= len(val_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))
		
		test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches" : len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": False, #optional
					"n_labels": 1}
		return data_objects
	if dataset_name =="stock50":
		train_X= torch.load(ROOT_PATH+'/0Stock_raw/50/train_X.pt')
		val_X = torch.load(ROOT_PATH+'/0Stock_raw/50/val_X.pt')
		test_X=torch.load(ROOT_PATH+'/0Stock_raw/50/test_X.pt')
		train_y= torch.load(ROOT_PATH+'/0Stock_raw/50/train_y.pt')
		val_y =torch.load(ROOT_PATH+'/0Stock_raw/50/val_y.pt')
		test_y= torch.load(ROOT_PATH+'/0Stock_raw/50/test_y.pt')
		
		seq_len = 24
		
		train_data = [] 
		val_data=[]
		test_data=[]
		
		for i in range(len(train_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			v = train_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()
			# import pdb ; pdb.set_trace()
			mask = check + zero.cuda()
			# mask = mask.cuda()
			labels = train_y[i,:,:].cuda()
			temp = (record_id,tt,vals.float(),mask,labels.float())
			train_data.append(temp)
		print("finish >> train data set")

		for i in range(len(val_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			v = val_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()
			mask = check + zero.cuda()
			
			labels = val_y[i,:,:].cuda()
			temp = (record_id,tt,vals.float(),mask,labels.float())
			val_data.append(temp)
		print("finish >> val data set")


		for i in range(len(test_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			v = test_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()
			mask = check + zero.cuda()
			
			labels = test_y[i,:,:].cuda()
			temp = (record_id,tt,vals.float(),mask,labels.float())
			test_data.append(temp)
		print("finish >> test data set")

		# import pdb ; pdb.set_trace()
		total_dataset = train_data+val_data+test_data 
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 
		batch_size = min(min(len(train_data), args.batch_size), args.n)
		
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		val_dataloader = DataLoader(val_data, batch_size= len(val_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))
		
		test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches" : len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": False, #optional
					"n_labels": 1}
		return data_objects	
	if dataset_name =="stock70":
		train_X= torch.load(ROOT_PATH+'/0Stock_raw/70/train_X.pt')
		val_X = torch.load(ROOT_PATH+'/0Stock_raw/70/val_X.pt')
		test_X=torch.load(ROOT_PATH+'/0Stock_raw/70/test_X.pt')
		train_y= torch.load(ROOT_PATH+'/0Stock_raw/70/train_y.pt')
		val_y =torch.load(ROOT_PATH+'/0Stock_raw/70/val_y.pt')
		test_y= torch.load(ROOT_PATH+'/0Stock_raw/70/test_y.pt')
		
		seq_len = 24
		
		train_data = [] 
		val_data=[]
		test_data=[]
		
		for i in range(len(train_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			v = train_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()
			# import pdb ; pdb.set_trace()
			mask = check + zero.cuda()
			# mask = mask.cuda()
			labels = train_y[i,:,:].cuda()
			temp = (record_id,tt,vals.float(),mask,labels.float())
			train_data.append(temp)
		print("finish >> train data set")

		for i in range(len(val_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			v = val_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()
			mask = check + zero.cuda()
			
			labels = val_y[i,:,:].cuda()
			temp = (record_id,tt,vals.float(),mask,labels.float())
			val_data.append(temp)
		print("finish >> val data set")


		for i in range(len(test_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(seq_len)).cuda()
			v = test_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()
			mask = check + zero.cuda()
			
			labels = test_y[i,:,:].cuda()
			temp = (record_id,tt,vals.float(),mask,labels.float())
			test_data.append(temp)
		print("finish >> test data set")

		# import pdb ; pdb.set_trace()
		total_dataset = train_data+val_data+test_data 
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 
		batch_size = min(min(len(train_data), args.batch_size), args.n)
		
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		val_dataloader = DataLoader(val_data, batch_size= len(val_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))
		
		test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches" : len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": False, #optional
					"n_labels": 1}
		return data_objects	



	if dataset_name == "activity":
		
		n_samples =  min(10000, args.n)
		dataset_obj = PersonActivity('data/PersonActivity', 
							download=True, n_samples =  n_samples, device = device)
		print(dataset_obj)
		# Use custom collate_fn to combine samples with arbitrary time observations.
		# Returns the dataset along with mask and time steps

		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(dataset_obj, train_size= 0.8, 
			random_state = 42, shuffle = True)

		train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
		test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

		record_id, tt, vals, mask, labels = train_data[0]
		input_dim = vals.size(-1)

		batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {"dataset_obj": dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": True, #optional
					"n_labels": labels.size(-1)}

		return data_objects

	########### 1d datasets ###########
# Physionet dataset

	if dataset_name == "sepsisiono":
		
		train_X= torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/NOIO/train_X.pt')
		val_X = torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/val_X.pt')
		test_X=torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/test_X.pt')
		train_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/train_y.pt')
		val_y =torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/val_y.pt')
		test_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/test_y.pt')
		# import pdb ; pdb.set_trace()
		# Combine and shuffle samples from physionet Train and physionet Test
		total_dataset = train_dataset_obj[:len(train_dataset_obj)]
		# import pdb ; pdb.set_trace()
		
		# import pdb ; pdb.set_trace()
		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, 
			random_state = 42, shuffle = True)

		record_id, tt, vals, mask, labels = train_data[0]
		
		
				
		# import pdb ; pdb.set_trace()
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 

		batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
		data_min, data_max = get_data_min_max(total_dataset)

		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max))
		test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))

		attr_names = train_dataset_obj.params
		data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"attr": attr_names, #optional
					"classif_per_tp": False, #optional
					"n_labels": 1} #optional
		return data_objects


	if dataset_name == "sepsisio":
		# import pdb ; pdb.set_trace()
		train_X= torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/train_X.pt')
		val_X = torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/val_X.pt')
		test_X=torch.load('/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/test_X.pt')
		train_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/train_y.pt')
		val_y =torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/val_y.pt')
		test_y= torch.load( '/home/bigdyl/socar/ACE_NODE/latent_ode/0SEPSIS/IO/test_y.pt')
		# import pdb ; pdb.set_trace()
		train_data = [] 
		val_data=[]
		test_data=[]
		
		for i in range(len(train_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(72)).cuda()
			v = train_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			labels = train_y[i]
			# if train_y[i] == 0:

			# 	labels = torch.zeros([72,1]).cuda()
			# else:
			# 	labels=torch.ones([72,1]).cuda()
			
			temp = (record_id,tt,vals,mask,labels)
			train_data.append(temp)
		print("finish >> train data set")
		for i in range(len(val_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(72)).cuda()
			v = val_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			labels = val_y[i]
			# if val_y[i] == 0:

			# 	labels = torch.zeros([72,1]).cuda()
			# else:
			# 	labels=torch.ones([72,1]).cuda()
			
			temp = (record_id,tt,vals,mask,labels)
			val_data.append(temp)
		print("finish >> val data set")
		for i in range(len(test_X)):
			record_id = str(i) 
			tt = torch.Tensor(np.arange(72)).cuda()
			v = test_X[i,:,:].cuda()
			check = ~torch.isnan(v) 
			zero = torch.zeros(check.shape)
			mask = check + zero.cuda()
			vals = torch.tensor(np.nan_to_num(np.array(v.cpu()))).cuda()	
			labels = test_y[i]
			# if test_y[i] == 0:

			# 	labels = torch.zeros([72,1]).cuda()
			# else:
			# 	labels=torch.ones([72,1]).cuda()
			
			temp = (record_id,tt,vals,mask,labels)
			test_data.append(temp)
		print("finish >> test data set")

		# Combine and shuffle samples from physionet Train and physionet Test
		
		
		total_dataset = train_data+val_data+test_data 
		n_samples = len(total_dataset)
		input_dim = vals.size(-1) # feature 
		batch_size = min(min(len(train_data), args.batch_size), args.n)
		
		data_min, data_max = get_data_min_max(total_dataset)
		
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max))
		val_dataloader = DataLoader(val_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))
		test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))

		
		data_objects = {"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader":utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": False, #optional
					"n_labels": 1} #optional
		return data_objects


	# Sampling args.timepoints time points in the interval [0, args.max_t]
	# Sample points for both training sequence and explapolation (test)
	distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
	time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
	time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
	time_steps_extrap = torch.sort(time_steps_extrap)[0]

	dataset_obj = None
	##################################################################
	# Sample a periodic function
	if dataset_name == "periodic":
		dataset_obj = Periodic_1d(
			init_freq = None, init_amplitude = 1.,
			final_amplitude = 1., final_freq = None, 
			z0 = 1.)

	##################################################################

	if dataset_obj is None:
		raise Exception("Unknown dataset: {}".format(dataset_name))

	dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n, 
		noise_weight = args.noise_weight)

	# Process small datasets
	dataset = dataset.to(device)
	time_steps_extrap = time_steps_extrap.to(device)

	train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

	n_samples = len(dataset)
	input_dim = dataset.size(-1)

	batch_size = min(args.batch_size, args.n)
	train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
	test_dataloader = DataLoader(test_y, batch_size = args.n, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))
	
	data_objects = {#"dataset_obj": dataset_obj, 
				"train_dataloader": utils.inf_generator(train_dataloader), 
				"test_dataloader": utils.inf_generator(test_dataloader),
				"input_dim": input_dim,
				"n_train_batches": len(train_dataloader),
				"n_test_batches": len(test_dataloader)}

	return data_objects


