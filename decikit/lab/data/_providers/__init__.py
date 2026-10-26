"""Named datasets: where each lives and how to read it."""
from decikit.lab.data._providers.base import DatasetError, DatasetProvider
from decikit.lab.data._providers.catalog import dataset, dataset_documents, datasets, datasets_folder, project_root
from decikit.lab.data._providers.files import FileProvider, FolderProvider

__all__ = ["DatasetError", "DatasetProvider", "FileProvider", "FolderProvider", "dataset", "dataset_documents", "datasets", "datasets_folder",
           "project_root"]
