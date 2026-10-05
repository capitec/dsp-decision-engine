"""Named datasets: where each lives and how to read it."""
from decider.data.providers.base import DatasetError, DatasetProvider
from decider.data.providers.catalog import dataset, dataset_documents, datasets, datasets_folder, project_root
from decider.data.providers.files import FileProvider, FolderProvider

__all__ = ["DatasetError", "DatasetProvider", "FileProvider", "FolderProvider", "dataset", "dataset_documents", "datasets", "datasets_folder",
           "project_root"]
