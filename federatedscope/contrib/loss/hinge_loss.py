from federatedscope.register import register_criterion
import torch


class HingeLoss(torch.nn.Module):
    
    def __init__(self, margin = 0.9):
        super(HingeLoss, self).__init__()
        self.margin = margin
    
    def forward(self, logits, target_item):
        """
        Args:
            logits : torch.Tensor
                The output of the model
            target_item : torch.Tensor
                The target item
        """
        positive_score = torch.gather(logits, 1, target_item.unsqueeze(1))
        loss = torch.max(torch.zeros_like(positive_score), self.margin - positive_score)
        
        return loss.mean()


def call_hinge_loss(type, device):
    try:
        import torch.nn as nn
    except ImportError:
        nn = None
        criterion = None

    if type == 'hinge':
        if nn is not None:
            criterion = HingeLoss().to(device)
        return criterion
    

register_criterion('hinge', call_hinge_loss)



class HingeWithSpredoutRegularization(torch.nn.Module):
    
    def __init__(self, margin = 0.9, spreadout_lambda = 0.1):
        super(HingeWithSpredoutRegularization, self).__init__()
        self.margin = margin
        self.spreadout_lambda = spreadout_lambda
    
    def forward(self, logits, target_item, model : torch.nn.Module):
        """
        Args:
            logits : torch.Tensor
                The output of the model
            target_item : torch.Tensor
                The target item
        """
        positive_score = torch.gather(logits, 1, target_item.unsqueeze(1))
        loss = torch.max(torch.zeros_like(positive_score), self.margin - positive_score)
        
        spreadout_loss = torch.norm(logits, p=2, dim=1).mean()
        
        return loss.mean() + self.spreadout_lambda * spreadout_loss