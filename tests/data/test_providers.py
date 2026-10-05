import json
import typing as t

import polars as pl
import pytest

from decider.data import DatasetError, DatasetProvider, FileProvider, FolderProvider
from decider.data._keys import DuplicateRecordKeyError

FRAME = pl.DataFrame({"client_id": [1, 2], "amount": [10, 20]})


@pytest.mark.parametrize("name,write", [
    ("a.csv", lambda p: FRAME.write_csv(p)),
    ("a.parquet", lambda p: FRAME.write_parquet(p)),
    ("a.jsonl", lambda p: FRAME.write_ndjson(p)),
    ("a.json", lambda p: p.write_text(FRAME.write_json())),
])
def test_file_provider_reads_each_format(tmp_path, name, write):
    write(tmp_path / name)
    assert FileProvider(path=str(tmp_path / name)).load().frame.equals(FRAME)


def test_file_without_a_known_suffix_is_an_error(tmp_path):
    with pytest.raises(DatasetError, match="format"):
        FileProvider(path=str(tmp_path / "a.dat")).scan()


def test_file_format_can_be_given(tmp_path):
    FRAME.write_csv(tmp_path / "a.dat")
    assert FileProvider(path=str(tmp_path / "a.dat"), format="csv").load().row_count == 2


def test_schema_and_head(tmp_path):
    FRAME.write_csv(tmp_path / "a.csv")
    provider = FileProvider(path=str(tmp_path / "a.csv"))
    assert provider.schema() == {"client_id": "Int64", "amount": "Int64"}
    assert provider.head(1).height == 1


def test_folder_of_parquet_parts_exposes_hive_partition_as_column(tmp_path):
    for region in ("za", "ke"):
        (tmp_path / f"region={region}").mkdir()
        FRAME.write_parquet(tmp_path / f"region={region}" / "part-0.parquet")
    frame = FolderProvider(path=str(tmp_path)).load().frame
    assert sorted(frame["region"].unique().to_list()) == ["ke", "za"]
    assert frame.height == 4


def test_folder_of_csv_parts_is_concatenated(tmp_path):
    FRAME.write_csv(tmp_path / "p1.csv")
    FRAME.write_csv(tmp_path / "p2.csv")
    (tmp_path / ".DS_Store").write_text("x")
    assert FolderProvider(path=str(tmp_path)).load().row_count == 4


def test_folder_of_json_parts_reads_ndjson(tmp_path):
    FRAME.write_ndjson(tmp_path / "p1.jsonl")
    FRAME.write_ndjson(tmp_path / "p2.jsonl")
    assert FolderProvider(path=str(tmp_path)).load().row_count == 4


def test_mixed_formats_in_a_folder_is_an_error(tmp_path):
    FRAME.write_csv(tmp_path / "a.csv")
    FRAME.write_parquet(tmp_path / "b.parquet")
    with pytest.raises(DatasetError, match="mixed formats.*csv.*parquet|mixed formats.*parquet.*csv"):
        FolderProvider(path=str(tmp_path)).scan()


def test_empty_folder_is_an_error(tmp_path):
    (tmp_path / ".gitkeep").write_text("")
    with pytest.raises(DatasetError, match="no data files"):
        FolderProvider(path=str(tmp_path)).scan()


def test_remote_folder_without_format_is_an_error():
    with pytest.raises(DatasetError, match="format"):
        FolderProvider(path="s3://bucket/loans").scan()


def test_duplicate_key_fails_on_load(tmp_path):
    pl.DataFrame({"id": [1, 1]}).write_csv(tmp_path / "a.csv")
    with pytest.raises(DuplicateRecordKeyError):
        FileProvider(path=str(tmp_path / "a.csv"), key=("id",)).load()


def test_key_is_the_suggested_key(tmp_path):
    FRAME.write_csv(tmp_path / "a.csv")
    assert FileProvider(path=str(tmp_path / "a.csv"), key=("amount",)).load().suggested_key == ("amount",)


def test_load_records_dataset_name_and_fingerprint(tmp_path):
    FRAME.write_csv(tmp_path / "a.csv")
    provider = FileProvider(path=str(tmp_path / "a.csv"))
    provider._name = "loans"
    data = provider.load()
    assert data.dataset == "loans"
    assert data.fingerprint


def test_subclass_fingerprint_is_used(tmp_path):
    class Fixed(FileProvider):
        type: t.Literal["fixed"] = "fixed"

        def fingerprint(self, frame):
            return "v42"

    FRAME.write_csv(tmp_path / "a.csv")
    assert Fixed(path=str(tmp_path / "a.csv")).load().fingerprint == "v42"


def test_builtin_aliases_resolve_without_importing_the_module():
    assert DatasetProvider.resolve("folder") is FolderProvider


def test_json_array_file(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"x": 1}, {"x": 2}]))
    assert FileProvider(path=str(tmp_path / "a.json")).load().row_count == 2
