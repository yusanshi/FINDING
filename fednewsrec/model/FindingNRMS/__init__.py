from fednewsrec.model.general.trainer.federated_group import FederatedGroupModel
from fednewsrec.model.NRMS import _NRMS


class FindingNRMS(FederatedGroupModel, _NRMS):
    # The inheritance order matters, see Python Method Resolution Order
    pass
