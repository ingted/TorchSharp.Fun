#r "nuget: FsPickler"
#r "nuget: FAkka.TorchSharp.DGX, 26.1.0-py3.8" 
#load "../../TorchSharp.Fun.DGX/TorchSharp.Fun.DGX.fs"

open TorchSharp.Fun.DGX
open TorchSharp

let device = if torch.cuda_is_available() then torch.CUDA else failwith "this test needs cuda"

let m1 =
    let l1 = torch.nn.Linear(10,10)
    let d = torch.nn.Dropout()
    let buf = new Modules.Parameter( torch.ones([|10L;10L|]),requires_grad=false)
    buf.name <- "buf"
    let bufRef = ref buf
    F [] [l1; d; bufRef] (fun t -> (t --> l1 --> d) + bufRef.Value)

let m2 = 
    torch.nn.Linear(10,10)
    ->> torch.nn.Linear(10,10)
    ->> torch.nn.ReLU()

let m = m1 ->> m2

m.Module.``to``(device)

let t1input = torch.rand([|10L|]).``to``(device)
let t' = m.forward t1input
