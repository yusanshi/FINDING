import torch

from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder
from fednewsrec.model.general.base import BaseModel
from fednewsrec.model.general.click_predictor.dot_product import DotProductClickPredictor
from fednewsrec.model.general.trainer.centralized import CentralizedModel
from fednewsrec.shared import args


class _NRMS(torch.nn.Module, BaseModel):
    """
    NRMS network.
    """

    def __init__(self, pretrained_word_embedding=None):
        super().__init__()
        self.news_encoder = NewsEncoder(pretrained_word_embedding)
        self.user_encoder = UserEncoder()
        self.click_predictor = DotProductClickPredictor()

    def forward(self, minibatch, news_pattern):
        """
        Args:

        Returns:
          click_probability: batch_size, 1 + K
        """
        single_news_length = list(news_pattern.values())[-1][-1]
        history = minibatch['history'].view(-1, single_news_length)
        positive_candidates = minibatch['positive_candidates']
        negative_candidates = minibatch['negative_candidates'].view(
            -1, single_news_length)

        vector = self.news_encoder(
            torch.cat((history, positive_candidates, negative_candidates),
                      dim=0))
        news_dim = vector.shape[-1]
        history_vector, positive_candidates_vector, negative_candidates_vector = vector.split(
            (history.shape[0], positive_candidates.shape[0],
             negative_candidates.shape[0]),
            dim=0)

        history_vector = history_vector.view(-1, args.num_history, news_dim)
        positive_candidates_vector = positive_candidates_vector.view(
            -1, 1, news_dim)
        negative_candidates_vector = negative_candidates_vector.view(
            -1, args.negative_sampling_ratio, news_dim)
        # batch_size, 1 + K, news_dim
        candidates_vector = torch.cat(
            (positive_candidates_vector, negative_candidates_vector), dim=1)

        # batch_size, news_dim
        user_vector = self.user_encoder(history_vector)
        # batch_size, 1 + K
        click_probability = self.click_predictor(
            candidates_vector,
            user_vector.unsqueeze(dim=1).expand_as(candidates_vector))
        return click_probability


class NRMS(CentralizedModel, _NRMS):
    pass
