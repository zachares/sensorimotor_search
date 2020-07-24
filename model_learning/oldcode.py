# "/scr-ssd/sens_search_logging/models/20200601_training_12_fit_wexpanded_action/"
  # Distance_Predictor:
  #   train: 1
  #   model_folder: ""
  #   epoch_num: 0
  #   inputs:
  #     macro_action: "dataset"
  #     tool_type: "dataset"
  #     option_type: "dataset"
  #     permutation: "Options_Sensor"
  #     epoch: "dataset"
  #   outputs:
  #     distances:
  #       loss: "MSE"
  #       loss_name: "Distance_Prediction"
  #       weight: 1
  #       inputs:
  #         distances: "Options_Sensor"
  #       evals: ["Continuous_Error"]
  # Options_ConfMat:
  #   train: 1
  #   model_folder: ""
  #   epoch_num: 0
  #   inputs:
  #     peg_type: "dataset"
  #     option_type: "dataset"
  #     options_est: "Options_Sensor"
  #     epoch: "dataset"


  # Vision_Sensor:
  #   train: 1
  #   model_folder: ""
  #   epoch: 0
  #   inputs:
  #     reference_image: dataset
  #     reference_depth: dataset
  #     reference_point_cloud: dataset
  #     state_idx: dataset
  #     point_cloud_point_ests: dataset
  #   outputs:
  #     pos_ests:
  #       inputs:
  #         hole_sites: dataset
  #       losses:
  #         L1:
  #           logging_name: Pos_Est_Vision
  #       evals:
  #         Continuous_Error:
  #           logging_name: Pos_Err_Vision
  #     heat_map_logits:
  #       inputs: 
  #         heat_map_idx: dataset
  #       losses:
  #         Multinomial_NLL:
  #           logging_name: Heat_Map_Class
  #     estimated_point:
  #       inputs:
  #         hole_sites: dataset
  #       evals:
  #         Continuous_Error:
  #           logging_name: Pos_Err_Heatmap

  class Vision_Sensor(Proto_Macromodel):
    def __init__(self, model_name, init_args, device = None):
        super().__init__()

        self.device = device
        self.model_list = []
        self.ensemble_list = []

        self.image_size = init_args['image_size']
        self.num_states = init_args['num_states']
        self.dropout_prob = init_args['dropout_prob']

        self.img_enc_size = 64

        self.num_cl = 3

        self.flatten = nn.Flatten()

        self.uc = True

        self.vision_list = []

        self.reference_position = torch.from_numpy(np.array([[0.5, 0.0, 0.89]])).to(device).float()

        for i in range(self.num_states):
            self.vision_list.append((\
                CONV2DN(model_name + "_image_enc" + str(i),\
                 (self.image_size[0], self.image_size[1], self.image_size[2]), (self.img_enc_size, 1, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = dropout_prob,\
                   uc = False, device = self.device).to(self.device),\

                CONV2DN(model_name + "_depth_enc" + str(i),\
                 (1, self.image_size[1], self.image_size[2]), (self.img_enc_size, 1, 1),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = dropout_prob,\
                   uc = False, device = self.device).to(self.device),\

                DECONV2DN(model_name + "_vision_dec" + str(i),\
                 (2 * self.img_enc_size, 2, 2), (2, self.image_size[1], self.image_size[2]),\
                  nonlinear = False, batchnorm = True, dropout = True, dropout_prob = dropout_prob,\
                   uc = False, device = self.device).to(self.device),\
            
                ResNetFCN(model_name + "_pos_est" + str(i),\
                2 * self.img_enc_size, 2, self.num_cl, dropout = True, dropout_prob = dropout_prob, \
                uc = False, device = self.device).to(self.device)))

        for model_tuple in self.vision_list:
            for model in model_tuple:
                self.model_list.append(model)

    def vision_outputs(self, input_dict, model_tuple, i):
        img_encoder, depth_encoder, vision_decoder, pos_estimator = model_tuple

        img_enc = self.flatten(img_encoder(input_dict['reference_image'].transpose(3,2).transpose(2,1)))
        depth_enc = self.flatten(depth_encoder(input_dict['reference_depth'].unsqueeze(1)))

        # pixels = input_dict['pixels'].long()
        # print(pixels.size())
        # u, v = pixels[0,0,0], pixels[0,0,1]

        # print(u, v)
        # print(input_dict['reference_point_cloud'].size())
        # print(input_dict['reference_point_cloud'][0,u,v])
        # print(input_dict['reference_point_cloud'][0,u,v].size())
        # print(input_dict['reference_point_cloud'].transpose(3,2).transpose(2,1)[0,:,u,v])
        # print(input_dict['reference_point_cloud'].transpose(3,2).transpose(2,1)[0,:,u,v].size())

        decoder_input = torch.cat([img_enc, depth_enc], dim = 1).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)

        heat_map_logits = vision_decoder(decoder_input)

        heat_map = F.softmax(heat_map_logits, dim = 1)

        heat_map_maxes = heat_map[:,1].view(heat_map.size(0), -1).max(1)[0]

        heat_map_mask = torch.where(heat_map[:,1] == heat_map_maxes.unsqueeze(1).unsqueeze(2).repeat(1,heat_map.size(2), heat_map.size(3)),\
            torch.ones_like(heat_map[:,1]), torch.zeros_like(heat_map[:,1]))

        estimated_point = (input_dict['reference_point_cloud'] * heat_map_mask.unsqueeze(3).repeat_interleave(3, dim=3)).sum(1).sum(1)\
         - self.reference_position.repeat_interleave(img_enc.size(0), dim = 0)

        # print((input_dict['reference_point_cloud'] * heat_map_mask.unsqueeze(3).repeat_interleave(3, dim=3)).sum(1).sum(1)[:10])

        # print(input_dict['point_cloud_point_ests'].size())
        # print(pos_estimator(torch.cat([img_enc, pc_enc, input_dict['point_cloud_point_ests'][:,i]], dim = 1)).size())
        pos_est = pos_estimator(torch.cat([img_enc, depth_enc], dim = 1)) / 10.0

        # print(pos_est[:10])

        return pos_est, heat_map_logits, estimated_point[:,:2]

    def forward(self, input_dict):
        pos_est_list = []
        heat_map_logits_list = []
        estimated_point_list = []

        for i, model_tuple in enumerate(self.vision_list):
            pos_est, heat_map_logits, estimated_point = self.vision_outputs(input_dict, model_tuple, i)

            pos_est_list.append(pos_est.unsqueeze(1))
            heat_map_logits_list.append(heat_map_logits.unsqueeze(2))
            estimated_point_list.append(estimated_point.unsqueeze(1))

        pos_ests = torch.cat(pos_est_list, dim = 1)

        return {
            'pos_ests': pos_ests,
            'heat_map_logits': torch.cat(heat_map_logits_list, dim = 2),
            'estimated_point': torch.cat(estimated_point_list, dim = 1),
        }

    def img_pos_estimate(self, input_dict):
        output_dict = self.forward(input_dict)

        return output_dict['pos_est'] + self.reference_position