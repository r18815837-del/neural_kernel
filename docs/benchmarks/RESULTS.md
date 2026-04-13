## LayerNorm Forward
- batch size: `32`
- normalized shape: `512`
- warmup: 3
- runs: 10
- mean: `0.00008104 s`
- median: `0.00008055 s`
- min: `0.00007780 s`
- max: `0.00008470 s`

## LayerNorm Backward
- batch size: `32`
- normalized shape: `512`
- warmup: 3
- runs: 10
- mean: `0.00026163 s`
- median: `0.00022350 s`
- min: `0.00020730 s`
- max: `0.00063710 s`

## MultiHeadAttention Forward
- batch size: `8`
- seq len: `32`
- d_model: `128`
- num_heads: `4`
- warmup: 3
- runs: 10
- mean: `0.00166366 s`
- median: `0.00157570 s`
- min: `0.00132770 s`
- max: `0.00254130 s`

## MultiHeadAttention Backward
- batch size: `8`
- seq len: `32`
- d_model: `128`
- num_heads: `4`
- warmup: 3
- runs: 10
- mean: `0.00837714 s`
- median: `0.00797655 s`
- min: `0.00709440 s`
- max: `0.01231970 s`

## TransformerBlock Forward
- batch size: `8`
- seq len: `32`
- d_model: `128`
- num_heads: `4`
- d_ff: `256`
- warmup: 3
- runs: 10
- mean: `0.00320950 s`
- median: `0.00318625 s`
- min: `0.00300440 s`
- max: `0.00354800 s`

## TransformerBlock Backward
- batch size: `8`
- seq len: `32`
- d_model: `128`
- num_heads: `4`
- d_ff: `256`
- warmup: 3
- runs: 10
- mean: `0.01560746 s`
- median: `0.01465265 s`
- min: `0.01378710 s`
- max: `0.02055470 s`