import argparse
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
from Models import modelpool
from Models.spike_temporal_adjust import SPIKE_SCHEDULE_MODES
from Preprocess import datapool
from utils import seed_all, get_torch_device

def calculate_fi_for_mode(args, train_loader, model, device, mode):
    print(f"\n>>> Starting FI calculation for mode: {mode}, T={args.time}")
    if hasattr(model, "set_spike_schedule"):
        model.set_spike_schedule(mode)
    
    fisher_per_timestep = []
    
    for t in range(1, args.time + 1):
        precision_matrices = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                precision_matrices[n] = torch.zeros_like(p.data)

        for step, (inputs, targets) in enumerate(train_loader):
            if step >= args.num_steps:
                break
            
            model.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)

            # SNN forward pass
            outputs = model(inputs) # Shape [T, B, num_classes]
            
            # Use mean of output up to time t for loss calculation
            avg_output = outputs[:t].mean(dim=0)

            loss = F.cross_entropy(avg_output, targets)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / args.num_steps
        
        # Calculate trace (sum of diag of FIM)
        fisher_trace = 0
        for n in precision_matrices:
            fisher_trace += precision_matrices[n].sum()

        trace_val = float(fisher_trace.detach().cpu())
        fisher_per_timestep.append(trace_val)
        print(f"[{mode}] Timestep {t}: FI Trace = {trace_val:.6f}")

    return fisher_per_timestep

def main():
    parser = argparse.ArgumentParser(description="Calculate Fisher Information for QCFS SNN")
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("-data", "--dataset", default="mnist", type=str, help="dataset")
    parser.add_argument("-arch", "--model", default="cnn2", type=str, help="model architecture")
    parser.add_argument("-T", "--time", default=8, type=int, help="SNN simulation time steps")
    parser.add_argument("-w", "--weights", required=True, type=str, help="path to model weights")
    parser.add_argument("-dev", "--device", default="auto", type=str, help="device")
    parser.add_argument("--num_steps", default=100, type=int, help="number of batches per FI calc")
    parser.add_argument(
        "--spike_schedule",
        default="normal",
        type=str,
        choices=sorted(SPIKE_SCHEDULE_MODES) + ["all"],
        help="Spike scheduling mode",
    )
    args = parser.parse_args()

    seed_all(44)
    device = get_torch_device(args.device)
    print(f"Using device: {device}")

    # Load data
    train_loader, _ = datapool(args.dataset, args.batch_size, num_workers=4)

    # Load model
    model = modelpool(args.model, dataset_name=args.dataset)
    checkpoint = torch.load(args.weights, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    from Models.cnn_mnist import remap_legacy_cnn2_state_dict
    from Models.VGG import remap_legacy_vgg_state_dict
    if args.model.startswith("cnn2"):
        state_dict, _ = remap_legacy_cnn2_state_dict(state_dict)
    elif "vgg" in args.model.lower():
        state_dict, _ = remap_legacy_vgg_state_dict(state_dict)
        
    model.load_state_dict(state_dict, strict=False)
    model.set_T(args.time)
    model.to(device)
    model.eval()

    modes_to_test = []
    if args.spike_schedule == "all":
        modes_to_test = sorted(SPIKE_SCHEDULE_MODES)
    else:
        modes_to_test = [args.spike_schedule]

    results = {}
    for mode in modes_to_test:
        fi_vals = calculate_fi_for_mode(args, train_loader, model, device, mode)
        results[mode] = fi_vals
        
        # Save each result to a file
        output_file = f"fi_{args.dataset}_{args.model}_T{args.time}_{mode}.txt"
        with open(output_file, "w") as f:
            f.write(",".join(map(str, fi_vals)))
        print(f"Results for {mode} saved to {output_file}")

    print("\n--- Summary of Results ---")
    for mode, vals in results.items():
        print(f"{mode}: {vals}")

if __name__ == "__main__":
    main()
