from fednewsrec.model.general.click_predictor.dot_product import DotProductClickPredictor


class BaseModel:

    def get_news_vector(self, *args, **kwargs):
        # batch_size, news_dim
        return self.news_encoder(*args, **kwargs)

    def get_user_vector(self, *args, **kwargs):
        # batch_size, news_dim
        return self.user_encoder(*args, **kwargs)

    @staticmethod
    def get_prediction(news_vector, user_vector):
        """
        The predictor used for model evaluation, will be pickled and passed
        to subprocesses (see `scoring_worker_fn` in `test.py`).

        Args:
            news_vector: candidate_size, news_dim
            user_vector: news_dim
        Returns:
            click_probability: candidate_size

        Note the `@staticmethod` and reinstantiated `DotProductClickPredictor()`,
        this is for avoiding the whole model object to be pickled with python multiprocessing,
        thus slowing down the code and potentially raising a "There appear to be %d leaked
        semaphore objects to clean up at shutdown" error.
        """
        # TODO: only used for dot-product clicker
        # candidate_size
        return DotProductClickPredictor()(news_vector,
                                          user_vector.expand_as(news_vector))
