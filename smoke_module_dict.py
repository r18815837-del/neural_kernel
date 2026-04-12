from kernel.nn.layers.linear import Linear
from kernel.nn.dropout import Dropout
from kernel.nn.modules.container import ModuleDict


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_module_dict_basic():
    mods = ModuleDict({
        "proj": Linear(8, 16),
        "drop": Dropout(0.1),
    })

    assert len(mods) == 2
    assert "proj" in mods
    assert "drop" in mods
    assert mods["proj"] is not None
    assert mods["drop"] is not None


def smoke_module_dict_setitem():
    mods = ModuleDict()
    mods["proj"] = Linear(8, 16)
    mods["head"] = Linear(16, 3)

    assert len(mods) == 2
    assert "proj" in mods
    assert "head" in mods


def smoke_module_dict_items():
    mods = ModuleDict({
        "proj": Linear(8, 16),
        "drop": Dropout(0.1),
    })

    keys = list(mods.keys())
    values = list(mods.values())
    items = list(mods.items())

    assert keys == ["proj", "drop"]
    assert len(values) == 2
    assert len(items) == 2
    assert items[0][0] == "proj"
    assert items[1][0] == "drop"


def smoke_module_dict_registered():
    mods = ModuleDict({
        "proj": Linear(8, 16),
        "drop": Dropout(0.1),
    })

    assert "proj" in mods._modules
    assert "drop" in mods._modules


def smoke_module_dict_invalid_key():
    failed = False
    try:
        mods = ModuleDict()
        mods[123] = Linear(8, 16)
    except TypeError:
        failed = True

    assert failed, "Expected TypeError for non-string key"


def smoke_module_dict_invalid_value():
    failed = False
    try:
        mods = ModuleDict()
        mods["bad"] = 123
    except TypeError:
        failed = True

    assert failed, "Expected TypeError for non-Module value"


def main():
    check("module dict basic", smoke_module_dict_basic)
    check("module dict setitem", smoke_module_dict_setitem)
    check("module dict items", smoke_module_dict_items)
    check("module dict registered", smoke_module_dict_registered)
    check("module dict invalid key", smoke_module_dict_invalid_key)
    check("module dict invalid value", smoke_module_dict_invalid_value)


if __name__ == "__main__":
    main()