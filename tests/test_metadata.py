import importlib


def test_package_author_metadata():
    pkg = importlib.import_module("sandgraph")
    assert hasattr(pkg, "__author__"), "__author__ must exist in package"
    expected = "Dong Liu, Yanxuan Yu, Ying Nian Wu, Xuhong Wang"
    assert pkg.__author__ == expected


def test_setup_metadata_strings_present():
    # Lightweight check that setup.py includes updated author lines
    with open("setup.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Dong Liu, Yanxuan Yu, Ying Nian Wu, Xuhong Wang" in content
    assert "dong.liu.dl2367@yale.edu" in content
    assert "yy3523@columbia.edu" in content


