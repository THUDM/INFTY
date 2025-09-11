import torch
from infty import optim as infty_optim


def get_infty_optimizer(params, base_optimizer, model, args):
    name = args.get("inftyopt").lower()
    
    if name == "base":
        return infty_optim.InftyBaseOptimizer(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "c_flat" or name == "c_flat_plus":
        return infty_optim.C_Flat(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "gam":
        return infty_optim.GAM(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "gsam":
        return infty_optim.GSAM(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "sam":
        return infty_optim.SAM(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "looksam":
        return infty_optim.LookSAM(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif "zo" in name:
        print("=" * 80)
        print(f"\033[93m[Infty Warning]\033[0m Detected ZO-based optimizer '{name}', which requires replacing the base optimizer.\n"
        f"We strongly recommend loading the model with pretrained weights, as zeroth-order gradient estimation can lead to highly unstable training.")
        lr = base_optimizer.param_groups[0]['lr']
        if "sgd" in name:
            print(f"\033[93m[Infty Info]\033[0m Replacing base optimizer with \033[1mSGD\033[0m (lr={lr}, momentum=0.9, weight_decay=5e-4)")
            base_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                momentum=0.9,
                weight_decay=5e-4
            )
        elif "adam" in name:
            print(f"\033[93m[Infty Info]\033[0m Replacing base optimizer with \033[1mAdam\033[0m (lr={lr}, weight_decay=5e-4)")
            base_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=5e-4
            )
        else:
            raise ValueError(f"Unknown infty_optimizer: {name}")
        print("=" * 80)
        return infty_optim.ZeroFlow(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "forward_grad":
        return infty_optim.ZeroFlow(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "pcgrad":
        return infty_optim.PCGrad(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "gradvac":
        return infty_optim.GradVac(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "cagrad":
        return infty_optim.CAGrad(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "unigrad_fs":
        return infty_optim.UniGrad_FS(params=params, base_optimizer=base_optimizer, model=model, args=args)
    elif name == "ogd":
        return infty_optim.OGD(params=params, base_optimizer=base_optimizer, model=model, args=args)
    else:
        raise ValueError(f"Unknown infty_optimizer: {name}")

