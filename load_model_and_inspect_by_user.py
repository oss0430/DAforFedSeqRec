import torch



model_path = "../../../../data1/donghoon/FederatedScopeData/exp/ml-1m/hidden_size_64_CE_lr1-e4_epoch_20_shadow_round_30K.pt"
dataset_path = ""


def sort_user_by_loss(
    model : torch.nn.Module,
    user_dataloader : torch.utils.data.DataLoader,   
) -> dict:
    """
    Sort the users by their loss
    """
    user_loss = {}
    model.eval()
    for user_id, user_data in user_dataloader.items():
        user_loss[user_id] = model(user_data)
    return user_loss


def sort_user_by_metric(
    model : torch.nn.Module,
    user_dataloader : torch.utils.data.DataLoader,
    metric : callable
) :
    """
    Sort the users by their metric
    """
    user_metric = {}
    model.eval()
    for user_id, user_data in user_dataloader.items():
        user_metric[user_id] = metric(model(user_data))
    return user_metric


def cluster_by_user_representation(
    model : torch.nn.Module,
    user_dataloader : torch.utils.data.DataLoader,
    cluster : callable
) :
    """
    Cluster the users by their representation
    """
    model.eval()


def cluster_by_logits(
    model : torch.nn.Module,
    user_dataloader : torch.utils.data.DataLoader,
    cluster : callable
) :
    """
    Cluster the users by their logits
    """
    model.eval()


def main():
    model = torch.load(model_path)
    print(model)
    
    
if __name__ == "__main__":
    main()
