from fednewsrec.model.general.trainer.federated_group import FederatedGroupModel
from fednewsrec.model.NAML import _NAML


class FindingNAML(FederatedGroupModel, _NAML):
    # The inheritance order matters, see Python Method Resolution Order
    pass
