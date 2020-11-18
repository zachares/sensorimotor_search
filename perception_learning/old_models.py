class Voting_Policy(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__(model_name)

        self.device = device
        self.model_list = []
        self.ensemble_list = []
        self.model_name = model_name

        self.num_tools = init_args['num_objects']
        self.num_states = init_args['num_objects']
        self.num_obs = 2
        self.num_objects = init_args['num_objects']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.state_size, device= self.device).to(self.device),\

            ))

        self.voting_function = mm.Transformer_Comparer(model_name + "_voting_function",\
          2 * self.state_size, 2, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device)

        self.reference_embeds = mm.Embedding(model_name + "_reference_vectors",\
                    self.num_tools, self.state_size, device= self.device).to(self.device)

        self.model_list.append(self.reference_embeds)
        self.model_list.append(self.voting_function)

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, shape_embed = model_tuple #origin_cov = model_tuple # spatial_processor,

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2)

        if 'trans_comparer' in seq_processor.model_name: # Transformer Padded
            if "padding_mask" in input_dict.keys():
                seq_enc = seq_processor(states_t.transpose(0,1), padding_mask = input_dict["padding_mask"]).max(0)[0]
            else:
                seq_enc = seq_processor(states_t.transpose(0,1)).max(0)[0]
            
        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        return states_T
                    
    def change_tool(self, encs, tool_idxs):
        frc_enc, seq_processor, shape_embed = self.ensemble_list[0] #origin_cov = model_tuple # spatial_processor,

        tool_embeds = shape_embed(tool_idxs)
        new_encs = encs.clone()
        new_encs[:, self.state_size:] = tool_embeds
        return new_encs

    def forward(self, input_dict):
        self.process_inputs(input_dict)

        enc = self.get_outputs(input_dict, self.ensemble_list[0])

        num_cand = random.choice([2,3,4,5,6,7,8])

        # idxs0 = torch.arange(input_dict['batch_size'])

        enc_comparison = enc.unsqueeze(1).repeat_interleave(num_cand, dim = 1)

        fit_idxs = (input_dict['fit_idx'] == 0).nonzero().squeeze()
        not_fit_idxs = (input_dict['fit_idx'] == 1).nonzero().squeeze()

        f_length = fit_idxs.size(0)
        nf_length = not_fit_idxs.size(0)

        assert f_length > 0

        assert nf_length > 0

        if f_length > nf_length:
            f_length = f_length
            nf_length = nf_length

            f_nf_idxs = fit_idxs[torch.randperm(f_length)]
            f_nf_idxs = f_nf_idxs[:nf_length]

            # print('Fit Indices ', fit_idxs.size())

            # print('Not Fit Indices ', not_fit_idxs.size())
            # print('Shuffled Fit Indices ', f_nf_idxs.size())

            nf_f_idxs = fit_idxs.clone()

            iterations = int(f_length / nf_length) + 1

            for k in range(iterations):
                if k < (iterations - 1):
                    nf_f_idxs[(k * nf_length):((k + 1) * nf_length)] = not_fit_idxs[torch.randperm(nf_length)]
                else:
                    extra_length = nf_f_idxs[k * nf_length:].size(0)
                    extra_labels = not_fit_idxs[torch.randperm(nf_length)]
                    nf_f_idxs[-extra_length:] = extra_labels[:extra_length]

            # print("Difference ", f_length - nf_length)
            # print("Shuffled Not Fit Indices", nf_f_idxs[:nf_length].size())
            # print("Additional indices not fit ", extra_idxs.size())

        elif f_length < nf_length:
            nf_f_idxs = not_fit_idxs[torch.randperm(nf_length)]
            nf_f_idxs = nf_f_idxs[:f_length]

            f_nf_idxs = not_fit_idxs.clone()

            iterations = int(nf_length / f_length) + 1

            for k in range(iterations):
                if k < (iterations - 1):
                    f_nf_idxs[(k * f_length):((k + 1) * f_length)] = fit_idxs[torch.randperm(f_length)]
                else:
                    extra_length = f_nf_idxs[k * f_length:].size(0)
                    extra_labels = fit_idxs[torch.randperm(f_length)]
                    f_nf_idxs[-extra_length:] = extra_labels[:extra_length]

            # print("Difference ", nf_length - f_length)

            # print(f_length)
            # print(extra_length)
            # print(extra_labels[:extra_length].size())
            # print(f_nf_idxs[f_length:].size())

        else:
            f_nf_idxs = fit_idxs[torch.randperm(f_length)]
            nf_f_idxs = not_fit_idxs[torch.randperm(nf_length)]

        idx_list = list(range(num_cand))

        random.shuffle(idx_list)

        frc_enc, seq_processor, shape_embed = self.ensemble_list[0] #origin_cov = model_tuple # spatial_processor,

        reference_vectors = torch.cat([ self.reference_embeds(input_dict['tool_idx']),\
         shape_embed(input_dict['tool_idx'])], dim = 1)

        mask = F.dropout(torch.ones((input_dict['batch_size'], num_cand)).float().to(self.device), p = 0.2)
        mask = torch.where(mask == 0, torch.zeros_like(mask), torch.ones_like(mask))

        voting_labels = torch.zeros((input_dict['batch_size'], num_cand)).long().to(self.device)

        mask_exp = mask.unsqueeze(2).repeat_interleave(2 * self.state_size, dim = 2)

        for i, j in enumerate(idx_list):
            mask_j = mask_exp[:,j]
            if i == 0:
                enc_comparison[:,j] = mask_j * enc_comparison[:,j] + (1 - mask_j) * reference_vectors
            elif i == 1:
                enc_to_shuffle = enc_comparison[:,j]
                enc_shuffled = enc_to_shuffle.clone()

                enc_shuffled[fit_idxs] = mask_j[fit_idxs] * enc_to_shuffle[nf_f_idxs] +\
                 (1 - mask_j[fit_idxs]) * reference_vectors[nf_f_idxs]

                enc_shuffled[not_fit_idxs] = mask_j[not_fit_idxs] * enc_to_shuffle[f_nf_idxs] +\
                 (1 - mask_j[not_fit_idxs]) * reference_vectors[f_nf_idxs]

                enc_comparison[:,j] = enc_shuffled

                voting_labels_to_shuffle = voting_labels[:,i]
                voting_labels_shuffled = voting_labels_to_shuffle.clone()
                voting_labels_shuffled[fit_idxs] = input_dict['fit_idx'][nf_f_idxs]
                voting_labels_shuffled[not_fit_idxs] = input_dict['fit_idx'][f_nf_idxs]

                voting_labels[:,j] = voting_labels_shuffled

            else:
                shuffle_idxs = torch.randperm(input_dict['batch_size'])

                if random.choice([0,1]) == 1:
                    # print('Before: ', enc_comparison[:, i])
                    enc_to_shuffle = enc_comparison[:,j]
                    enc_shuffled = mask_j * enc_to_shuffle[shuffle_idxs] + (1 - mask_j) * reference_vectors[shuffle_idxs]
                    enc_comparison[:,j] = enc_shuffled
                    # print("Set: ", enc_shuffled )
                    # print("After: ", enc_comparison[:, i])
                    voting_labels[:,j] = input_dict['fit_idx'][shuffle_idxs]
                else:
                    new_reference_vectors = torch.cat([self.reference_embeds(input_dict['new_tool_idx']),\
                     shape_embed(input_dict['new_tool_idx'])], dim = 1)
                    
                    enc_to_shuffle = self.change_tool(enc_comparison[:,j], input_dict['new_tool_idx'])
                    enc_shuffled = mask_j * enc_to_shuffle[shuffle_idxs] + (1 - mask_j) * new_reference_vectors[shuffle_idxs]
                    enc_comparison[:,j] = enc_shuffled          
                    voting_labels[:,j] = input_dict['new_fit_idx'][shuffle_idxs]

        # voting_labels = voting_labels * mask + (1 - mask) * random_votes
        tool_state_embeddings = self.voting_function(enc_comparison.transpose(0,1))
        fit_logits = (tool_state_embeddings[:,:,:self.state_size] * tool_state_embeddings[:,:,self.state_size:]).sum(2).transpose(0,1)
        voting_probs = F.softmax(fit_logits, dim = 1)

        # print(voting_labels)

        not_fit_probs = torch.where(voting_labels == 1, voting_probs, torch.zeros_like(voting_probs)).sum(1).unsqueeze(1) + 0.01
        fit_probs = torch.where(voting_labels == 0, voting_probs, torch.zeros_like(voting_probs)).sum(1).unsqueeze(1) + 0.01

        # print(not_fit_probs)
        # print(fit_probs)

        vote_logits = torch.log(torch.cat([fit_probs, not_fit_probs], dim = 1))

        # print(vote_logits)

        vote_labels = torch.zeros(input_dict['batch_size']).to(self.device).long()

        # print(state_logits_unc[:10])
        # print(input_dict['done_mask'][:10])
        # print(state_logits[:10])
        # print(input_dict['done_mask'][:,0].mean())

        # print((1 / pos_post_var)[:10])
        # if torch.isnan(1 / pos_post_var).any():
        #     print("Problem")

        return {
            'vote_logits': vote_logits,
            'vote_inputs': multinomial.logits2inputs(vote_logits),
            'vote_idx': vote_labels,
        }

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def candidate_weights(self, input_dict, other_objects):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)
            # input_dict['state_idx'] = input_dict['tool_idx']

            pos_ests_mean, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            return int(obs_idx.item()), obs_state_logprobs.squeeze().cpu().numpy()

    def process_inputs(self, input_dict):
        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):
        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['state_idx'] = input_dict['hole_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

        if 'rel_pos_init' in input_dict.keys():
            input_dict['reference_pos'] = 100 * input_dict['rel_pos_init'].unsqueeze(0).repeat_interleave(T, 0)

class History_Encoder_Baseline(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_class" + str(i),\
            self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))


        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, shape_embed, pos_estimator, obs_classifier = model_tuple #origin_cov = model_tuple # spatial_processor,

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_enc = seq_processor(states_t).max(0)[0]

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests = pos_estimator(states_T)

        obs_logits = obs_classifier(states_T)

        return pos_ests, obs_logits, states_T

    def forward(self, input_dict):
        self.process_inputs(input_dict)
        pos_ests, obs_logits, enc = self.get_outputs(input_dict, self.ensemble_list[0])
        
        return {
            'pos_est': pos_ests,
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
        }      

    def get_encoding(self, input_dict):
        with torch.no_grad():
            self.eval()
            ### need self uc to be off
            # prev_time = time.time()          
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests_obs, obs_logits, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            return enc

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)
        
class Variational_History_Encoder(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_params_est" + str(i),\
            self.state_size, 2 * self.state_size, 2, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_class" + str(i),\
            self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, params_estimator, shape_embed, pos_estimator, obs_classifier = model_tuple #origin_cov = model_tuple # spatial_processor,

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_output = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_output = seq_processor(states_t).max(0)[0]

        seq_params = params_estimator(seq_output)

        seq_mean, seq_var  = gaussian_parameters(seq_params)

        seq_enc = sample_gaussian(seq_mean, seq_var, self.device)

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests = pos_estimator(states_T)

        obs_logits = obs_classifier(states_T)

        return pos_ests, obs_logits, torch.cat([seq_mean, tool_embed], dim = 1), seq_mean, seq_var

    def forward(self, input_dict):
        self.process_inputs(input_dict)
        pos_ests, obs_logits, enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])
        
        return {
            'pos_est': pos_ests,
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
            'enc_params': (enc_mean, 1 / enc_var),
            'prior_params': (torch.zeros_like(enc_mean), torch.ones_like(enc_var)),
        }      

    def get_encoding(self, input_dict):
        with torch.no_grad():
            self.eval()
            ### need self uc to be off
            # prev_time = time.time()          
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests, obs_logits, enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])

            return enc

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

class Selfsupervised_History_Encoder(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_params_est" + str(i),\
            self.state_size, 2 * self.state_size, 2, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_pred" + str(i),\
            self.state_size + self.tool_dim + self.action_size, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_force_pred" + str(i),\
            self.state_size + self.tool_dim + self.action_size, 6, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_contact_pred" + str(i),\
            self.state_size + self.tool_dim + self.action_size, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pairing_class" + str(i),\
            self.state_size, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, params_estimator, shape_embed,\
         pos_predictor, force_predictor, contact_predictor, pairing_classifier = model_tuple

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_output = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_output = seq_processor(states_t).max(0)[0]

        seq_mean, seq_var  = gaussian_parameters(params_estimator(seq_output))

        seq_enc = sample_gaussian(seq_mean, seq_var, self.device)

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        action_states_T = torch.cat([seq_enc, tool_embed, input_dict['final_action']], dim = 1)

        pos_preds = pos_predictor(action_states_T)

        force_preds = force_predictor(action_states_T)

        contact_preds = contact_predictor(action_states_T)

        paired_logits = pairing_classifier(seq_enc)

        if 'force_unpaired_reshaped' in input_dict.keys():
            frc_encs_unpaired_reshaped = self.flatten(frc_enc(input_dict["force_unpaired_reshaped"]))

            frc_unpaired_encs = torch.reshape(frc_encs_unpaired_reshaped,\
             (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

            states_unpaired_t = torch.cat([frc_unpaired_encs, input_dict['sensor_inputs_unpaired']], dim = 2).transpose(0,1)

            if "padding_mask" in input_dict.keys():
                seq_unpaired_output = seq_processor(states_unpairedno_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
            else:
                seq_unpaired_output = seq_processor(states_unpaired_t).max(0)[0]

            seq_unpaired_mean, seq_unpaired_var  = gaussian_parameters(params_estimator(seq_unpaired_output))

            seq_unpaired_enc = sample_gaussian(seq_unpaired_mean, seq_unpaired_var, self.device)

            unpaired_logits = pairing_classifier(seq_unpaired_enc)

        else:
            unpaired_logits = None

        return pos_preds, force_preds, contact_preds, paired_logits, unpaired_logits, torch.cat([seq_mean, tool_embed], dim = 1), seq_mean, seq_var

    def forward(self, input_dict):
        self.process_inputs(input_dict)
        pos_preds, force_preds, contact_preds, paired_logits, unpaired_logits,\
        enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])

        paired_idxs = torch.cat([
            torch.ones_like(paired_logits[:,0]).long(), torch.zeros_like(unpaired_logits[:,0]).long()
            ], dim = 0)

        paired_combined_logits = torch.cat([paired_logits, unpaired_logits], dim = 0)
        
        return {
            'pos_pred': pos_preds,
            'force_pred': force_preds,
            'contact_pred': contact_preds,
            'contact_inputs': multinomial.logits2inputs(contact_preds),
            'paired_class': paired_combined_logits,
            'paired_inputs': multinomial.logits2inputs(paired_combined_logits),
            'paired_idx': paired_idxs,
            'unpaired_inputs': multinomial.logits2inputs(paired_logits),
            'enc_params': (enc_mean, 1 / enc_var),
            'prior_params': (torch.zeros_like(enc_mean), torch.ones_like(enc_var)),
        }      

    def get_encoding(self, input_dict):
        with torch.no_grad():
            self.eval()
            ### need self uc to be off
            # prev_time = time.time()          
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_preds, force_preds, contact_preds, paired_logits, unpaired_logits,\
            enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])

            return enc

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool
    
    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['proprio_diff'] = torch.where(input_dict['proprio'][:,1:] != 0,\
         input_dict['proprio'][:,1:] - input_dict['proprio'][:,:-1], torch.zeros_like(input_dict['proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['proprio_diff']], dim = 2)

        if 'force_hi_freq_unpaired' in input_dict.keys():
            input_dict['force_unpaired'] = input_dict['force_hi_freq_unpaired'].transpose(2,3)

            input_dict['force_unpaired_reshaped'] = torch.reshape(input_dict["force_unpaired"],\
         (input_dict["force_unpaired"].size(0) * input_dict["force_unpaired"].size(1), \
         input_dict["force_unpaired"].size(2), input_dict["force_unpaired"].size(3)))

            input_dict['proprio_unpaired_diff'] = torch.where(input_dict['proprio_unpaired'][:,1:] != 0,\
             input_dict['proprio_unpaired'][:,1:] - input_dict['proprio_unpaired'][:,:-1], torch.zeros_like(input_dict['proprio_unpaired'][:,1:]))            
            
            input_dict['sensor_inputs_unpaired'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['proprio_unpaired_diff']], dim = 2)

    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["proprio"] = input_dict['proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)
        input_dict['final_action'] = input_dict['action'][:,-1].repeat_interleave(T,0)

class Unsupervised_History_Encoder(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []
        self.model_name = model_name

        self.num_tools = init_args['num_objects']
        self.num_states = init_args['num_objects']
        self.num_obs = init_args['num_objects']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_params_est" + str(i),\
            self.state_size, 2 * self.state_size, 2, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_force_est" + str(i),\
            self.state_size + self.tool_dim, 6, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_contact_class" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, params_estimator, shape_embed,\
         pos_estimator, force_estimator, contact_classifier = model_tuple

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_output = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_output = seq_processor(states_t).max(0)[0]

        seq_params = params_estimator(seq_output)

        seq_mean, seq_var  = gaussian_parameters(seq_params)

        seq_enc = sample_gaussian(seq_mean, seq_var, self.device)

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests = pos_estimator(states_T)

        force_ests = force_estimator(states_T)

        contact_logits = contact_classifier(states_T)

        return pos_ests, force_ests, contact_logits, torch.cat([seq_mean, tool_embed], dim = 1), seq_mean, seq_var

    def forward(self, input_dict):
        self.process_inputs(input_dict)
        pos_ests, force_ests, contact_logits, enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])
        
        return {
            'pos_est': pos_ests,
            'force_est': force_ests,
            'contact_class': contact_logits,
            'contact_inputs': multinomial.logits2inputs(contact_logits),
            'enc_params': (enc_mean, 1 / enc_var),
            'prior_params': (torch.zeros_like(enc_mean), torch.ones_like(enc_var)),
        }      

    def get_encoding(self, input_dict):
        with torch.no_grad():
            self.eval()
            ### need self uc to be off
            # prev_time = time.time()          
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests, force_ests, contact_logits, enc, enc_mean, enc_var = self.get_outputs(input_dict, self.ensemble_list[0])

            return enc

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['proprio_diff'] = torch.where(input_dict['proprio'][:,1:] != 0,\
         input_dict['proprio'][:,1:] - input_dict['proprio'][:,:-1], torch.zeros_like(input_dict['proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["proprio"] = input_dict['proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)

class StatePosSensor_wUncertainty(mm.Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []
        self.model_name = model_name

        self.num_tools = init_args['num_tools']
        self.num_states = init_args['num_states']
        self.num_obs = init_args['num_states']

        self.action_size = init_args['action_size']
        self.force_size = init_args['force_size']
        self.proprio_size = init_args['proprio_size']
        self.min_length = init_args['min_length']
        self.contact_size = 1

        self.dropout_prob = init_args['dropout_prob']

        self.force_mean = torch.from_numpy(np.array(init_args['force_mean'])).to(self.device).float()
        self.force_std = torch.from_numpy(np.array(init_args['force_std'])).to(self.device).float()

        self.frc_enc_size = 48
        self.tool_dim = 6

        self.state_size = self.frc_enc_size + self.proprio_size + self.action_size + self.contact_size

        self.num_ensembles = 1

        self.num_tl = 4
        self.num_cl = 3
        self.flatten = nn.Flatten()
        self.uc = False

        for i in range(self.num_ensembles):
            self.ensemble_list.append((\
                mm.CONV1DN(model_name + "_frc_enc" + str(i),\
                 (self.force_size[0], self.force_size[1]), (self.frc_enc_size, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = self.dropout_prob,\
                   uc = self.uc, device = self.device).to(self.device),\

                mm.Transformer_Comparer(model_name + "_sequential_process" + str(i),\
          self.state_size, self.num_tl, dropout_prob = self.dropout_prob, uc = self.uc, device = self.device).to(self.device),\
                
                mm.Embedding(model_name + "_shape_embed" + str(i),\
                         self.num_tools, self.tool_dim, device= self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est_dyn_noise" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_pos_est_obs_noise" + str(i),\
            self.state_size + self.tool_dim, 2, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_class" + str(i),\
            self.state_size + self.tool_dim, self.num_obs, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\

                mm.ResNetFCN(model_name + "_obs_likelihood" + str(i),\
            self.state_size + self.tool_dim, self.num_obs * self.num_states, self.num_cl, dropout = True, dropout_prob = self.dropout_prob, \
            uc = self.uc, device = self.device).to(self.device),\
            ))

        for model_tuple in self.ensemble_list:
            for model in model_tuple:
                self.model_list.append(model)

    def get_outputs(self, input_dict, model_tuple, final_idx = None):
        frc_enc, seq_processor, shape_embed, pos_estimator, dyn_noise_estimator, obs_noise_estimator,\
         obs_classifier, obs_likelihood = model_tuple #origin_cov = model_tuple # spatial_processor,

        ##### Calculating Series Encoding
        frc_encs_reshaped = self.flatten(frc_enc(input_dict["force_reshaped"]))

        frc_encs = torch.reshape(frc_encs_reshaped,\
         (input_dict['batch_size'], input_dict['sequence_size'], self.frc_enc_size))

        states_t = torch.cat([frc_encs, input_dict['sensor_inputs']], dim = 2).transpose(0,1)

        if "padding_mask" in input_dict.keys():
            seq_enc = seq_processor(states_t, padding_mask = input_dict["padding_mask"]).max(0)[0]
        else:
            seq_enc = seq_processor(states_t).max(0)[0]

        tool_embed = shape_embed(input_dict['tool_idx'].long()) 

        states_T = torch.cat([seq_enc, tool_embed], dim = 1)

        pos_ests_obs = pos_estimator(states_T)
        
        pos_ests_dyn_noise = dyn_noise_estimator(states_T).pow(2) + 1e-2 #+ pos_ests_obs.pow(2)

        pos_ests_obs_noise = obs_noise_estimator(states_T).pow(2) + 1e-2

        obs_logits = obs_classifier(states_T)

        # likelihood network
        obs_state_logits = torch.reshape(obs_likelihood(states_T), (input_dict['batch_size'], self.num_obs, self.num_states))
        obs_state_logprobs = F.log_softmax(obs_state_logits, dim = 1)

        state_logprobs = obs_state_logprobs[torch.arange(input_dict['batch_size']), F.softmax(obs_logits, dim = 1).max(1)[1]]

        return pos_ests_obs, pos_ests_dyn_noise, pos_ests_obs_noise, obs_logits, state_logprobs, obs_state_logprobs, states_T

    def forward(self, input_dict):
        self.process_inputs(input_dict)

        pos_ests_mean, pos_ests_dyn_noise, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

        prior_noise = input_dict['rel_pos_prior_var'] #+ pos_ests_dyn_noise

        y = pos_ests_mean - input_dict['rel_pos_prior_mean']
        S = prior_noise + pos_ests_obs_noise
        K = prior_noise / S

        pos_post = input_dict['rel_pos_prior_mean'] + K * y
        pos_post_var = (1 - K) * prior_noise

        state_logits = torch.log(input_dict['state_prior']) + obs_logprobs

        # print(state_logits_unc[:10])
        # print(input_dict['done_mask'][:10])
        # print(state_logits[:10])
        # print(input_dict['done_mask'][:,0].mean())

        # print((1 / pos_post_var)[:10])
        # if torch.isnan(1 / pos_post_var).any():
        #     print("Problem")

        return {
            'pos_est': pos_post,
            'pos_est_params': (pos_post, 1 / pos_post_var),
            'obs_logits': obs_logits,
            'obs_inputs': multinomial.logits2inputs(obs_logits),
            'state_logits': state_logits,
            'state_inputs': multinomial.logits2inputs(state_logits),
            'obs_logprobs_inputs': multinomial.logits2inputs(obs_logprobs),
        }

    def set_uc(self, uc_bool):
        for model in self.ensemble_list[0]:
            if hasattr(model, 'set_uc'):
                model.set_uc(uc_bool)
            elif hasattr(model, 'uc'):
                model.uc = uc_bool

    def pos_params(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests_mean, pos_ests_dyn_noise, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            return 0.01 * pos_ests_mean.squeeze().cpu().numpy(), 0.0001 * pos_ests_obs_noise.squeeze().cpu().numpy()

    def type_params(self, input_dict):
        with torch.no_grad():
            self.eval()
            # prev_time = time.time()
            ### Method 1 #### without uncertainty estimation
            T = 1
            self.test_time_process_inputs(input_dict, T)
            self.process_inputs(input_dict)
            self.set_uc(False)

            pos_ests_mean,  pos_ests_dyn_noise, pos_ests_obs_noise, obs_logits, obs_logprobs, obs_state_logprobs, enc = self.get_outputs(input_dict, self.ensemble_list[0])

            obs_idx = F.softmax(obs_logits, dim = 1).max(1)[1]

            return int(obs_idx.item()), obs_state_logprobs.squeeze().cpu().numpy()
              
    def process_inputs(self, input_dict):

        ##### feature processing
        input_dict["force"] = input_dict["force_hi_freq"].transpose(2,3)
        
        input_dict["force_reshaped"] = torch.reshape(input_dict["force"],\
         (input_dict["force"].size(0) * input_dict["force"].size(1), \
         input_dict["force"].size(2), input_dict["force"].size(3)))

        input_dict["batch_size"] = input_dict["force"].size(0)
        input_dict["sequence_size"] = input_dict["force"].size(1)

        input_dict['rel_proprio_diff'] = torch.where(input_dict['rel_proprio'][:,1:] != 0,\
         input_dict['rel_proprio'][:,1:] - input_dict['rel_proprio'][:,:-1], torch.zeros_like(input_dict['rel_proprio'][:,1:]))

        input_dict['sensor_inputs'] = torch.cat([input_dict['action'], input_dict['contact'], input_dict['rel_proprio_diff']], dim = 2)
    
    def test_time_process_inputs(self, input_dict, T):

        input_dict["force_hi_freq"] = input_dict["force_hi_freq"][:,1:].repeat_interleave(T, 0)
        input_dict["action"] = input_dict["action"][:, :-1].repeat_interleave(T, 0)
        input_dict["rel_proprio"] = input_dict['rel_proprio'].repeat_interleave(T, 0)
        input_dict['tool_idx'] = input_dict['peg_vector'].max(0)[1].long().unsqueeze(0).repeat_interleave(T, 0)
        input_dict['contact'] = input_dict['contact'][:,1:].repeat_interleave(T,0)