import logging
import copy
import torch
from federatedscope.core.auxiliaries.utils import param2tensor
from federatedscope.core.aggregators import ClientsAvgAggregator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VarianceReportingClientAvgAggregator(ClientsAvgAggregator):
    
    def _para_weighted_avg(self, models, recover_fun=None):
        """
        Calculates the weighted average of models.
        """
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size
        
        sample_size, avg_model = models[0]
        model_variance = copy.deepcopy(avg_model)
        
        for key in avg_model:
            variance = 0.0
            mean = 0.0
            for i in range(len(models)):
                local_sample_size, local_model = models[i]

                if key not in local_model:
                    continue

                if self.cfg.federate.ignore_weight:
                    weight = 1.0 / len(models)
                elif self.cfg.federate.use_ss:
                    # When using secret sharing, what the server receives
                    # are sample_size * model_para
                    weight = 1.0
                else:
                    weight = local_sample_size / training_set_size

                if not self.cfg.federate.use_ss:
                    local_model[key] = param2tensor(local_model[key])
                
                # Welford's algorithm to update mean and variance 
                
                delta =  local_model[key] - mean
                mean += delta * weight
                delta_2 = local_model[key] - mean
                variance += float(torch.flatten(delta * delta_2).mean())
                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

            ## round off variance to 5 decimal places
            model_variance[key] = round(variance, 5)
            mean = 0.0
            variance = 0.0
            
            if self.cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                # When using secret sharing, what the server receives are
                # sample_size * model_para
                avg_model[key] /= training_set_size
                avg_model[key] = torch.FloatTensor(avg_model[key])

        model_variance = {"weight_variance": model_variance}
        logger.info(f"{model_variance}")
        
        return avg_model