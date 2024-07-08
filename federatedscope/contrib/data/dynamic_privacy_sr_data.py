import os
import numpy as np
import logging
import torch
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from federatedscope.contrib.data.sr_data import SequentialRecommendationDataset
from federatedscope.core.data import BaseDataTranslator
from federatedscope.register import register_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DynamicSRDataset(SequentialRecommendationDataset):
    
    def __init__(self, data_dir : str, data_name : str, data_type : str, data_size : str, **kwargs):
        super(DynamicSRDataset, self).__init__(data_dir, data_name, data_type, data_size)
        self.dynamic_privacy