from src.download_data import find_dataset_root


def test_find_dataset_root_nested_and_case_insensitive(tmp_path):
    root = tmp_path / "cache" / "v1" / "PCB_DATASET"
    (root / "Annotations" / "x").mkdir(parents=True)
    (root / "images" / "x").mkdir(parents=True)
    (root / "rotation").mkdir()
    assert find_dataset_root(tmp_path) == str(root)
    assert find_dataset_root(tmp_path / "missing") is None
    lower = tmp_path / "hf"
    (lower / "annotations").mkdir(parents=True); (lower / "images").mkdir()
    assert find_dataset_root(lower) == str(lower)
