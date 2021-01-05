import torch
from torch import nn
import numpy as np
from typing import Optional


def _reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == 'elementwise_mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f'{reduction} is not a valid reduction')


def cumulative_link_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                         reduction: str = 'elementwise_mean',
                         class_weights: Optional[np.ndarray] = None
                         ) -> torch.Tensor:
    eps = 1e-15
    likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true), eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        # Make sure it's on the same device as neg_log_likelihood
        class_weights = torch.as_tensor(class_weights,
                                        dtype=neg_log_likelihood.dtype,
                                        device=neg_log_likelihood.device)
        neg_log_likelihood *= class_weights[y_true]

    loss = _reduction(neg_log_likelihood, reduction)
    return loss


class CumulativeLinkLoss(nn.Module):

    def __init__(self, reduction: str = 'elementwise_mean',
                 class_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        return cumulative_link_loss(y_pred, y_true,
                                    reduction=self.reduction,
                                    class_weights=self.class_weights)

class LogisticCumulativeLink(nn.Module):

    def __init__(self, num_classes: int,
                 init_cutpoints: str = 'ordered') -> None:
        assert num_classes > 2, (
            'Only use this model if you have 3 or more classes'
        )
        super().__init__()
        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        if init_cutpoints == 'ordered':
            num_cutpoints = self.num_classes - 1
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
            self.cutpoints = nn.Parameter(cutpoints)
        elif init_cutpoints == 'random':
            cutpoints = torch.rand(self.num_classes - 1).sort()[0]
            self.cutpoints = nn.Parameter(cutpoints)
        else:
            raise ValueError(f'{init_cutpoints} is not a valid init_cutpoints '
                             f'type')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Equation (11) from
        "On the consistency of ordinal regression methods", Pedregosa et. al.
        """
        sigmoids = torch.sigmoid(self.cutpoints - X)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )
        return link_mat


class OrdinalLogisticModel(nn.Module):

    def __init__(self, n_classes: int,
                 init_cutpoints: str = 'ordered') -> None:
        super().__init__()
        self.num_classes = n_classes
        self.link = LogisticCumulativeLink(self.num_classes,
                                           init_cutpoints=init_cutpoints)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.link(X)
