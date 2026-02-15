# TorchSharp.Fun.DGX

A wrapper over `FAkka.TorchSharp.DGX` that provides a **function-compostion** style of model construction.
It keeps the same composition style (`->>`, `-->`, `F/Fx`, `IModel`) and adds DGX/NVFP4-oriented helpers.

## Example usage

```F#
let lisht() = 
    F [] [] (fun z ->
        use g = torch.nn.functional.tanh(z)
        z * g
    )

let model =
    torch.nn.EmbeddingBag(EMB_INDX_DIM,EMB_DIM)
    ->> torch.nn.Linear(EMB_DIM,BTL_N)
    ->> lisht()
    ->> torch.nn.Dropout(0.1)
    ->> torch.nn.Linear(BTL_N,BTL_N)
    ->> lisht()
    ->> torch.nn.Dropout(0.1)
    ->> torch.nn.Linear(BTL_N,BASE_TGTS)

```

## DGX / NVFP4 helpers

`TorchSharp.Fun.DGX` adds composable stages in `TorchSharp.Fun.DGX.DGX`:

```fsharp
open TorchSharp.Fun.DGX

let model =
    DGX.managedInputDefault()
    ->> DGX.nvfp4Input DGX.Float16
    ->> torch.nn.Linear(4096, 4096)
    ->> DGX.castInput DGX.Float32
```

- `DGX.managedInput()`:
  - best-effort UM promotion (`UnifiedMemory.tryPromoteToManaged`) for zero-copy-oriented flow.
- `DGX.nvfp4Input policy`:
  - quantize-dequantize adapter (`Nvfp4Training.quantizePacked/dequantizePacked`) for NVFP4 input path.
- `DGX.packNvfp4Sink policy output`:
  - packs FP output back to NVFP4 sink tensors (`qdata`, `scale`).
