
import os
from reliquery.storage import FileStorage
from reliquery import Relic


def _create_storage_names(relic_path):
    relic_path = os.path.normpath(relic_path)
    root = os.path.dirname(relic_path)
    relic_type = relic_path.split(os.sep)[-1]
    return root, relic_type


def _get_relic(relic_path, node_name):
    root, relic_type = _create_storage_names(relic_path)
    storage = FileStorage(root, node_name)
    relic = Relic(name=node_name, relic_type=relic_type, storage=storage)

    return relic


def get_parameters_df_from_relic(relic_path, node_name):
    relic = _get_relic(relic_path, node_name)
    return relic.get_pandasdf('Parameters')


def get_substate_df_from_relic(relic_path, node_name):
    relic = _get_relic(relic_path, node_name)
    return relic.get_pandasdf('Substate')

