import copy
import torch
import re
from federatedscope.core.aggregators import ClientsAvgAggregator


class SASRecAggregator(ClientsAvgAggregator):
    
    def __init__(self, model=None, device='cpu', config=None):
        super(SASRecAggregator, self).__init__(model, device, config)
        
        self.base_aggregation_rules = config.aggregator.BFT_args.base_aggregation_rules
        if self.base_aggregation_rules == "krum" or self.base_aggregation_rules == "median":
            self.base_byzantine_node_num = config.aggregator.BFT_args.base_byzantine_node_num
            self.sequence_encoder_byzantine_node_num = config.aggregator.BFT_args.sequence_encoder_byzantine_node_num
            if self.base_aggregation_rules == "krum" :
                self.base_krum_agg_num = config.aggregator.BFT_args.base_krum_agg_num
                self.sequence_encoder_krum_agg_num = config.aggregator.BFT_args.sequence_encoder_krum_agg_num
            
        elif self.base_aggregation_rules == "normbounding":
            self.base_normbound = config.aggregator.BFT_args.base_normbound
            self.sequence_encoder_norm_bound = config.aggregator.BFT_args.sequence_encoder_norm_bound
        
        elif self.base_aggregation_rules == "trimmedmean":
            self.base_excluded_ratio = config.aggregator.BFT_args.base_excluded_ratio
            self.sequence_encoder_excluded_ratio = config.aggregator.BFT_args.sequence_encoder_excluded_ratio
            self.base_byzantine_node_num = config.aggregator.BFT_args.base_byzantine_node_num
            self.sequence_encoder_byzantine_node_num = config.aggregator.BFT_args.sequence_encoder_byzantine_node_num
        
        all_model_keys = model.state_dict().keys()
        self.item_embedding_weight_keys = [key for key in all_model_keys if re.match(r'item_embedding', key)]
        self.position_embedding_weight_keys = [key for key in all_model_keys if re.match(r'position_embedding', key)]
        self.sequence_encoder_weight_keys = [key for key in all_model_keys if re.match(r'trm_encoder\.layer\.\d+\.', key)]
        #dict_keys(['item_embedding.weight', 'position_embedding.weight', 'trm_encoder.layer.0.multi_head_attention.query.weight', 'trm_encoder.layer.0.multi_head_attention.query.bias', 'trm_encoder.layer.0.multi_head_attention.key.weight', 'trm_encoder.layer.0.multi_head_attention.key.bias', 'trm_encoder.layer.0.multi_head_attention.value.weight', 'trm_encoder.layer.0.multi_head_attention.value.bias', 'trm_encoder.layer.0.multi_head_attention.dense.weight', 'trm_encoder.layer.0.multi_head_attention.dense.bias', 'trm_encoder.layer.0.multi_head_attention.LayerNorm.weight', 'trm_encoder.layer.0.multi_head_attention.LayerNorm.bias', 'trm_encoder.layer.0.feed_forward.dense_1.weight', 'trm_encoder.layer.0.feed_forward.dense_1.bias', 'trm_encoder.layer.0.feed_forward.dense_2.weight', 'trm_encoder.layer.0.feed_forward.dense_2.bias', 'trm_encoder.layer.0.feed_forward.LayerNorm.weight', 'trm_encoder.layer.0.feed_forward.LayerNorm.bias', 'trm_encoder.layer.1.multi_head_attention.query.weight', 'trm_encoder.layer.1.multi_head_attention.query.bias', 'trm_encoder.layer.1.multi_head_attention.key.weight', 'trm_encoder.layer.1.multi_head_attention.key.bias', 'trm_encoder.layer.1.multi_head_attention.value.weight', 'trm_encoder.layer.1.multi_head_attention.value.bias', 'trm_encoder.layer.1.multi_head_attention.dense.weight', 'trm_encoder.layer.1.multi_head_attention.dense.bias', 'trm_encoder.layer.1.multi_head_attention.LayerNorm.weight', 'trm_encoder.layer.1.multi_head_attention.LayerNorm.bias', 'trm_encoder.layer.1.feed_forward.dense_1.weight', 'trm_encoder.layer.1.feed_forward.dense_1.bias', 'trm_encoder.layer.1.feed_forward.dense_2.weight', 'trm_encoder.layer.1.feed_forward.dense_2.bias', 'trm_encoder.layer.1.feed_forward.LayerNorm.weight', 'trm_encoder.layer.1.feed_forward.LayerNorm.bias', 'LayerNorm.weight', 'LayerNorm.bias'])
            
            
    def aggregate(self, agg_info):
        """
        To perform aggregation seperately for each part of the model
        """
        models = agg_info["client_feedback"]
        #item_embeddings = [zip(each_model[0], each_model[1].item_embedding)  for each_model in models]
        #position_embeddings = [zip(each_model[0], each_model[1].position_embedding)  for each_model in models]
        #{k: dictionary[k] for k in keys if k in dictionary}
        sequence_encoder = []
        for each_model in models:
            model_weight = {}
            for key in each_model[1]:
                if key in self.sequence_encoder_weight_keys:
                    model_weight[key] = each_model[1][key]
            sequence_encoder.append((each_model[0], model_weight))
        
        if self.base_aggregation_rules == "krum":
            avg_model = self._para_avg_with_krum(models,
                                                 agg_num= self.base_krum_agg_num,
                                                 byzantine_node_num = self.base_byzantine_node_num)
            avg_seq_encoder = self._para_avg_with_krum(sequence_encoder,
                                                       agg_num= self.sequence_encoder_krum_agg_num,
                                                       byzantine_node_num = self.sequence_encoder_byzantine_node_num)
        elif self.base_aggregation_rules == "median":
            avg_model = self._aggre_with_median(models)
            avg_seq_encoder = self._aggre_with_median(sequence_encoder)
        elif self.base_aggregation_rules == "normbounding":
            avg_model = self._aggre_with_normbounding(models, self.base_normbound)
            avg_seq_encoder = self._aggre_with_normbounding(sequence_encoder, self.sequence_encoder_norm_bound)
        elif self.base_aggregation_rules == "trimmedmean":
            avg_model = self._aggre_with_trimmedmean(models, self.base_excluded_ratio)
            avg_seq_encoder = self._aggre_with_trimmedmean(sequence_encoder, self.sequence_encoder_excluded_ratio)
        
        
        update_model = copy.deepcopy(avg_model)
        init_model = self.model.state_dict()
        for key in avg_model:
            update_model[key] = init_model[key] + avg_model[key]
        
        for key in avg_seq_encoder:
            update_model[key] = init_model[key] + avg_seq_encoder[key]
            
        return update_model

    
    def _aggre_with_normbounding(self, models, normbound):
        models_temp = []
        for each_model in models:
            param, ignore_keys = self._flatten_updates(each_model[1])
            if torch.norm(param, p=2) > normbound:
                scaling_rate = normbound / torch.norm(param, p=2)
                scaled_param = scaling_rate * param
                models_temp.append(
                    (each_model[0],
                     self._reconstruct_updates(scaled_param, ignore_keys)))
            else:
                models_temp.append(each_model)
        return self._para_weighted_avg(models_temp)
    
    
    def _flatten_updates(self, model):
        model_update, ignore_keys = [], []
        init_model = self.model.state_dict()
        for key in init_model:
            if key not in model:
                ignore_keys.append(key)
                continue
            model_update.append(model[key].view(-1))
        return torch.cat(model_update, dim=0), ignore_keys


    def _reconstruct_updates(self, flatten_updates, ignore_keys):
        start_idx = 0
        init_model = self.model.state_dict()
        reconstructed_model = copy.deepcopy(init_model)
        for key in init_model:
            if key in ignore_keys:
                continue
            reconstructed_model[key] = flatten_updates[
                start_idx:start_idx + len(init_model[key].view(-1))].reshape(
                    init_model[key].shape)
            start_idx = start_idx + len(init_model[key].view(-1))
        return reconstructed_model
    
    
    def _calculate_distance(self, model_a, model_b):
        """
        Calculate the Euclidean distance between two given model para delta
        """
        distance = 0.0

        for key in model_a:
            if isinstance(model_a[key], torch.Tensor):
                model_a[key] = model_a[key].float()
                model_b[key] = model_b[key].float()
            else:
                model_a[key] = torch.FloatTensor(model_a[key])
                model_b[key] = torch.FloatTensor(model_b[key])

            distance += torch.dist(model_a[key], model_b[key], p=2)
        return distance

    def _calculate_score(self, models, byzantine_node_num):
        """
        Calculate Krum scores
        """
        model_num = len(models)
        closest_num = model_num - byzantine_node_num - 2

        distance_matrix = torch.zeros(model_num, model_num)
        for index_a in range(model_num):
            for index_b in range(index_a, model_num):
                if index_a == index_b:
                    distance_matrix[index_a, index_b] = float('inf')
                else:
                    distance_matrix[index_a, index_b] = distance_matrix[
                        index_b, index_a] = self._calculate_distance(
                            models[index_a], models[index_b])

        sorted_distance = torch.sort(distance_matrix)[0]
        krum_scores = torch.sum(sorted_distance[:, :closest_num], axis=-1)
        return krum_scores


    def _para_avg_with_krum(self, models, agg_num=1, byzantine_node_num = 0):

        # each_model: (sample_size, model_para)
        models_para = [each_model[1] for each_model in models]
        krum_scores = self._calculate_score(models_para, byzantine_node_num)
        index_order = torch.sort(krum_scores)[1].numpy()
        reliable_models = list()
        for number, index in enumerate(index_order):
            if number < agg_num:
                reliable_models.append(models[index])

        return self._para_weighted_avg(models=reliable_models)
    
    
    def _aggre_with_median(self, models):
        _, init_model = models[0]
        global_update = copy.deepcopy(init_model)
        for key in init_model:
            temp = torch.stack([each_model[1][key] for each_model in models],
                               0)
            temp_pos, _ = torch.median(temp, dim=0)
            temp_neg, _ = torch.median(-temp, dim=0)
            global_update[key] = (temp_pos - temp_neg) / 2
        return global_update
    
    
    def _aggre_with_trimmedmean(self, models, excluded_ratio):
        _, init_model = models[0]
        global_update = copy.deepcopy(init_model)
        excluded_num = int(len(models) * excluded_ratio)
        for key in init_model:
            temp = torch.stack([each_model[1][key] for each_model in models],
                               0)
            pos_largest, _ = torch.topk(temp, excluded_num, 0)
            neg_smallest, _ = torch.topk(-temp, excluded_num, 0)
            new_stacked = torch.cat([temp, -pos_largest,
                                     neg_smallest]).sum(0).float()
            new_stacked /= len(temp) - 2 * excluded_num
            global_update[key] = new_stacked
        return global_update
