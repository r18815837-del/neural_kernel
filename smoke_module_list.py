from kernel.nn.module import Module
from kernel.nn.modules.container import ModuleList
from kernel.nn.modules.transformer import TransformerBlock


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_module_list_basic():
    ml = ModuleList()
    ml.append(TransformerBlock(d_model=8, num_heads=2, d_ff=16))
    ml.append(TransformerBlock(d_model=8, num_heads=2, d_ff=16))

    assert len(ml) == 2
    assert isinstance(ml[0], Module)
    assert isinstance(ml[1], Module)


def smoke_module_list_iter():
    ml = ModuleList(
        [
            TransformerBlock(d_model=8, num_heads=2, d_ff=16),
            TransformerBlock(d_model=8, num_heads=2, d_ff=16),
        ]
    )

    count = 0
    for _ in ml:
        count += 1

    assert count == 2


def smoke_module_list_registered():
    ml = ModuleList(
        [
            TransformerBlock(d_model=8, num_heads=2, d_ff=16),
            TransformerBlock(d_model=8, num_heads=2, d_ff=16),
        ]
    )

    assert "0" in ml._modules
    assert "1" in ml._modules


def main():
    check("module list basic", smoke_module_list_basic)
    check("module list iter", smoke_module_list_iter)
    check("module list registered", smoke_module_list_registered)


if __name__ == "__main__":
    main()