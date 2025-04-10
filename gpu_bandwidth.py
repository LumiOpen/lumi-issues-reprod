import torch
import time
import numpy as np
import argparse
from tqdm import tqdm

def format_size(size_bytes):
    """Format size in bytes to a human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def measure_bandwidth(source_device, target_device, tensor_size, iterations=10, warmup=3):
    """Measure data transfer bandwidth between two devices"""
    # Create tensor on source device
    x = torch.rand(tensor_size, device=source_device)
    
    # Ensure target device is ready
    target = torch.empty(tensor_size, device=target_device)
    
    # Synchronize devices
    if source_device.startswith('cuda') or target_device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # Warmup
    for _ in range(warmup):
        target.copy_(x)
        if source_device.startswith('cuda') or target_device.startswith('cuda'):
            torch.cuda.synchronize()
    
    # Measure time
    start_time = time.time()
    for _ in range(iterations):
        target.copy_(x)
        if source_device.startswith('cuda') or target_device.startswith('cuda'):
            torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate bandwidth
    elapsed_time = end_time - start_time
    tensor_bytes = x.element_size() * x.nelement()
    total_bytes = tensor_bytes * iterations
    bandwidth = total_bytes / elapsed_time
    
    return bandwidth, tensor_bytes

def run_bandwidth_tests(sizes=None, iterations=10, warmup=3):
    """Run a series of bandwidth tests between available devices"""
    if sizes is None:
        # Default tensor sizes (in elements)
        sizes = [
            (1000, 1000),        # ~4 MB for float32
            (5000, 5000),        # ~100 MB for float32
            (10000, 10000),      # ~400 MB for float32
            (20000, 20000),      # ~1.6 GB for float32
        ]
    
    # Check available devices
    devices = ['cpu']
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        devices.append(f'cuda:{i}')
    
    print(f"Found {len(devices)} devices: {', '.join(devices)}")
    
    results = {}
    
    # Test all device pairs
    for source in devices:
        for target in devices:
            if source == target:
                continue
                
            pair_name = f"{source} → {target}"
            print(f"\nTesting {pair_name}")
            pair_results = []
            
            for size in sizes:
                tensor_size = size
                try:
                    bandwidth, tensor_bytes = measure_bandwidth(
                        source, target, tensor_size, iterations, warmup
                    )
                    bandwidth_gb_per_s = bandwidth / (1024**3)
                    pair_results.append({
                        'tensor_size': tensor_size,
                        'tensor_bytes': tensor_bytes,
                        'bandwidth_bytes_per_s': bandwidth,
                        'bandwidth_gb_per_s': bandwidth_gb_per_s
                    })
                    print(f"  Size {tensor_size}: {format_size(tensor_bytes)} - "
                          f"Bandwidth: {bandwidth_gb_per_s:.2f} GB/s")
                except RuntimeError as e:
                    print(f"  Size {tensor_size}: Error - {str(e)}")
            
            results[pair_name] = pair_results
    
    # Print summary
    print("\nBandwidth Summary (GB/s):")
    summary_table = {}
    
    for pair, measurements in results.items():
        if not measurements:
            continue
        source, target = pair.split(' → ')
        if source not in summary_table:
            summary_table[source] = {}
        avg_bandwidth = np.mean([m['bandwidth_gb_per_s'] for m in measurements])
        summary_table[source][target] = avg_bandwidth
    
    # Print table
    print("\n" + " " * 10, end="")
    for device in devices:
        print(f"{device:>10}", end="")
    print()
    
    for source in devices:
        print(f"{source:>10}", end="")
        for target in devices:
            if source == target:
                print(f"{'N/A':>10}", end="")
            elif target in summary_table.get(source, {}):
                print(f"{summary_table[source][target]:>10.2f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure GPU-CPU communication bandwidth')
    parser.add_argument('--iterations', type=int, default=10, 
                        help='Number of iterations for each test')
    parser.add_argument('--warmup', type=int, default=3, 
                        help='Number of warmup iterations')
    args = parser.parse_args()
    
    print("PyTorch Device Communication Bandwidth Test")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    run_bandwidth_tests(iterations=args.iterations, warmup=args.warmup)
