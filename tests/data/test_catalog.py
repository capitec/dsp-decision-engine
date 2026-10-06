import json
import typing as t

import polars as pl
import pytest

from decider.lab.data import DatasetError, DatasetProvider, FileProvider, FolderProvider, dataset, datasets
from decider.lab.data._providers import dataset_documents, datasets_folder, project_root

FRAME = pl.DataFrame({"client_id": [1, 2]})


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "datasets").mkdir()
    return tmp_path / "datasets"


def test_files_and_folders_are_inferred_by_name(folder):
    FRAME.write_csv(folder / "loans.csv")
    (folder / "parts").mkdir()
    FRAME.write_parquet(folder / "parts" / "p.parquet")
    (folder / ".hidden").mkdir()
    (folder / "notes.txt").write_text("x")
    found = datasets(path=str(folder))
    assert {k: type(v) for k, v in found.items()} == {"loans": FileProvider, "parts": FolderProvider}
    assert found["loans"].name == "loans"
    assert found["loans"].load().row_count == 2


def test_config_entry_overrides_an_inferred_dataset(folder):
    FRAME.write_csv(folder / "loans.csv")
    (folder / "config.json").write_text(json.dumps({"loans": {"type": "file", "path": "loans.csv", "key": ["client_id"]}}))
    assert datasets(path=str(folder))["loans"].key == ("client_id",)


def test_relative_config_path_resolves_against_the_datasets_folder(folder):
    FRAME.write_csv(folder / "raw.csv")
    (folder / "config.json").write_text(json.dumps({"x": {"type": "file", "path": "raw.csv"}}))
    assert datasets(path=str(folder))["x"].load().row_count == 2


def test_url_path_in_config_is_left_alone(folder):
    (folder / "config.json").write_text(json.dumps({"x": {"type": "file", "path": "s3://b/x.parquet"}}))
    assert dataset_documents(path=str(folder))["x"]["path"] == "s3://b/x.parquet"


def test_client_subclass_named_by_import_path_resolves(folder):
    (folder / "config.json").write_text(json.dumps({"w": {"type": f"{__name__}:Warehouse", "table": "t"}}))
    assert datasets(path=str(folder))["w"].load().row_count == 2


class Warehouse(DatasetProvider):
    type: t.Literal["warehouse"] = "warehouse"
    table: str

    def scan(self):
        return FRAME.lazy()


def test_name_clash_between_inferred_datasets_is_an_error(folder):
    FRAME.write_csv(folder / "loans.csv")
    (folder / "loans").mkdir()
    with pytest.raises(DatasetError, match="loans"):
        datasets(path=str(folder))


def test_same_stem_different_suffix_is_a_name_clash(folder):
    FRAME.write_csv(folder / "loans.csv")
    FRAME.write_parquet(folder / "loans.parquet")
    with pytest.raises(DatasetError, match="loans.csv.*loans.parquet"):
        datasets(path=str(folder))


def test_every_bad_config_entry_is_named_in_one_error(folder):
    (folder / "config.json").write_text(json.dumps({
        "a": {"type": "file"}, "b": {"type": "nope", "path": "x"}, "ok": {"type": "file", "path": "x.csv"}}))
    with pytest.raises(DatasetError) as info:
        datasets(path=str(folder))
    assert "a:" in str(info.value) and "b:" in str(info.value) and "ok:" not in str(info.value)


def test_documents_are_returned_unvalidated(folder):
    (folder / "config.json").write_text(json.dumps({"bad": {"type": "file"}}))
    assert dataset_documents(path=str(folder)) == {"bad": {"type": "file"}}


def test_missing_folder_gives_no_datasets(tmp_path):
    assert datasets(path=str(tmp_path / "nope")) == {}


def test_project_root_expands_from_a_start_file(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "datasets").mkdir()
    FRAME.write_csv(tmp_path / "datasets" / "loans.csv")
    assert project_root(tmp_path / "src" / "flow.py") == tmp_path.resolve()
    assert list(datasets(start=tmp_path / "src" / "flow.py")) == ["loans"]


def test_project_root_expands_from_the_cwd(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "datasets").mkdir()
    (tmp_path / "sub").mkdir()
    FRAME.write_csv(tmp_path / "datasets" / "loans.csv")
    monkeypatch.chdir(tmp_path / "sub")
    assert list(datasets()) == ["loans"]


def test_relative_path_is_relative_to_the_cwd(tmp_path, monkeypatch):
    (tmp_path / "datasets").mkdir()
    FRAME.write_csv(tmp_path / "datasets" / "loans.csv")
    monkeypatch.chdir(tmp_path)
    assert list(datasets(path="./datasets")) == ["loans"]


def test_datasets_path_comes_from_the_environment(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    FRAME.write_csv(tmp_path / "d" / "loans.csv")
    monkeypatch.setenv("Decider_DATASETS_PATH", str(tmp_path / "d"))
    assert list(datasets()) == ["loans"]


def test_one_dataset_loads_despite_a_broken_sibling(folder):
    FRAME.write_csv(folder / "loans.csv")
    (folder / "config.json").write_text(json.dumps({"bad": {"type": "file"}}))
    assert dataset("loans", path=str(folder)).name == "loans"
    with pytest.raises(DatasetError, match="invalid dataset 'bad'"):
        dataset("bad", path=str(folder))


def test_unknown_dataset_names_the_known_ones(folder):
    FRAME.write_csv(folder / "loans.csv")
    with pytest.raises(DatasetError, match="unknown dataset 'nope'.*loans"):
        dataset("nope", path=str(folder))


def test_datasets_folder_expands_the_project_root(tmp_path):
    (tmp_path / "configs").mkdir()
    assert datasets_folder(tmp_path / "x.py") == tmp_path.resolve() / "datasets"
