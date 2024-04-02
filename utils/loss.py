import torch


# They are using this really weird loss that I don't understand
class CrossEntropySurvivalLoss:
    def __call__(self, hazards, S, Y, c):
        return ce_loss(hazards, S, Y, c)


def ce_loss(hazards, S, Y, c, eps=1e-7):
    # batch_size is always 1
    batch_size = len(Y)
    # ground truth
    Y = Y.view(batch_size, 1)
    # censorship status, 0 or 1
    # always 0 because considering only dead patients in our case
    c = c.view(batch_size, 1).float()
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # This is L_uncensored
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    # Bho I don't get the second term
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = ce_l + reg
    loss = loss.mean()
    return loss
