import torch

def nll_loss(output, target, pos_weight=0.5, neg_weight=0.5):
    match_ids = (target[:, :-1, :-1] == 1).nonzero(as_tuple=True)
    unmatch1_ids = (target[:, :, -1] == 1).nonzero(as_tuple=True)
    unmatch2_ids = (target[:, -1, :] == 1).nonzero(as_tuple=True)
    
    pos_loss = -torch.mean(output[match_ids])
    neg_loss1 = -torch.mean(output[unmatch1_ids[0], unmatch1_ids[1], -1])
    neg_loss2 = -torch.mean(output[unmatch2_ids[0], -1, unmatch2_ids[1]])
    loss = pos_weight * pos_loss + neg_weight * (neg_loss1 + neg_loss2)
    
    return loss/target.size(0)


def nll_loss_le(output, target, unmatch_id, pos_weight=0.5, neg_weight=0.5):
    match_ids = ((target > 0) & (target < unmatch_id)).nonzero(as_tuple=False)
    match_ids = torch.transpose(match_ids, 1, 0)
    match_ids = torch.cat([match_ids, torch.unsqueeze(target[match_ids.numpy()], dim=0)], dim=0)
    match_probs = output[match_ids.numpy()]
    match_loss = -torch.sum(match_probs)/len(match_ids)

    unmatch_ids = (target == unmatch_id).nonzero(as_tuple=False)
    unmatch_ids = torch.transpose(unmatch_ids, 1, 0)
    unmatch_ids = torch.cat([unmatch_ids, torch.unsqueeze(target[unmatch_ids.numpy()], dim=0)], dim=0)
    unmatch_probs = output[unmatch_ids.numpy()]   
    unmatch_loss = -torch.sum(unmatch_probs)/len(unmatch_ids)

    loss = pos_weight * match_loss + neg_weight * unmatch_loss
    
    return loss/target.size(0)
