# Troubleshooting GPU Access in Podman

## Problem: Container can't access GPU

If you see errors like "GPU not accessible in containers", try these solutions in order:

### Solution 1: Verify Host GPU Access

```bash
# Check that GPU works on host
nvidia-smi

# Should show your GPU details without errors
```

If this fails, your NVIDIA drivers aren't working properly. Fix that first.

### Solution 2: Check Device Files

```bash
# Verify device files exist
ls -l /dev/nvidia*

# Should show:
# /dev/nvidia0
# /dev/nvidiactl
# /dev/nvidia-uvm
# /dev/nvidia-modeset
```

If missing, reboot or reload nvidia modules:
```bash
sudo modprobe nvidia
sudo modprobe nvidia-uvm
```

### Solution 3: Test CDI Configuration

```bash
# Check if CDI config exists
ls -l /etc/cdi/nvidia.yaml

# If missing, generate it
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Test CDI access
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.3.1-base-ubi9 nvidia-smi
```

### Solution 4: Use Direct Device Mapping

If CDI doesn't work, use direct device mapping instead:

```bash
# Run the alternative script
./run-direct-device.sh 4

# Or manually:
podman run --rm \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --security-opt=label=disable \
    nvidia/cuda:12.3.1-base-ubi9 nvidia-smi
```

### Solution 5: Check SELinux

SELinux can block GPU access. Temporarily test without it:

```bash
# Check SELinux status
getenforce

# Temporarily set to permissive
sudo setenforce 0

# Test GPU access
podman run --rm --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm nvidia/cuda:12.3.1-base-ubi9 nvidia-smi

# If it works, re-enable and use security-opt
sudo setenforce 1
podman run --rm --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --security-opt=label=disable nvidia/cuda:12.3.1-base-ubi9 nvidia-smi
```

### Solution 6: Check Permissions

```bash
# Check device permissions
ls -l /dev/nvidia*

# Add your user to video/render groups if needed
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Log out and back in for groups to take effect
```

### Solution 7: Rootless vs Rootful Podman

Try switching between rootless and rootful:

```bash
# Check if running rootless
podman info | grep -i rootless

# Try as root (if currently rootless)
sudo podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.3.1-base-ubi9 nvidia-smi

# Or with sudo for rootful mode
sudo ./benchmark.sh run
```

### Solution 8: NVIDIA Container Toolkit Configuration

```bash
# Verify toolkit is installed
rpm -qa | grep nvidia-container-toolkit

# Check nvidia-ctk version
nvidia-ctk --version

# Reconfigure and regenerate
sudo nvidia-ctk runtime configure --runtime=podman
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Restart Podman service (if rootful)
sudo systemctl restart podman
```

## Quick Diagnostic Script

Run this to diagnose issues:

```bash
#!/bin/bash
echo "=== GPU Diagnostics ==="
echo ""

echo "1. Host GPU access:"
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1

echo ""
echo "2. Device files:"
ls -l /dev/nvidia* 2>&1 | head -5

echo ""
echo "3. CDI configuration:"
ls -l /etc/cdi/nvidia.yaml 2>&1

echo ""
echo "4. User groups:"
id | grep -E '(video|render)'

echo ""
echo "5. SELinux status:"
getenforce 2>&1

echo ""
echo "6. Podman mode:"
podman info | grep -i rootless

echo ""
echo "7. NVIDIA Container Toolkit:"
rpm -qa | grep nvidia-container-toolkit

echo ""
echo "8. Test CDI:"
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.3.1-base-ubi9 nvidia-smi 2>&1 | head -5

echo ""
echo "9. Test direct devices:"
podman run --rm --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --security-opt=label=disable nvidia/cuda:12.3.1-base-ubi9 nvidia-smi 2>&1 | head -5
```

## Recommended Approach for Your Setup

Based on your output, the CDI was generated but Podman can't use it. Here's what to try:

### Option A: Fix CDI (Preferred)

```bash
# Ensure CDI file has correct permissions
sudo chmod 644 /etc/cdi/nvidia.yaml

# Verify CDI contents
grep "nvidia.com/gpu" /etc/cdi/nvidia.yaml

# Test with specific GPU
podman run --rm --device nvidia.com/gpu=0 nvidia/cuda:12.3.1-base-ubi9 nvidia-smi
```

### Option B: Use Direct Mapping (Immediate)

```bash
# Just use the direct device script
./run-direct-device.sh 4
```

This will work immediately and is perfectly fine for your use case. The benchmark doesn't care which method is used to access the GPU.

## Common RHEL 10 Beta Issues

If you're on RHEL 10 beta:

1. **Container toolkit may not be fully supported yet**
   - Fallback to direct device mapping is fine
   
2. **Package GPG keys**
   - You may need: `sudo rpm --import /etc/pki/rpm-gpg/RPM-GPG-KEY-*`

3. **Kernel updates**
   - After kernel updates, regenerate CDI: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`

## Bottom Line

**For immediate use**: Just use `./run-direct-device.sh 4` which bypasses CDI entirely and directly maps GPU devices. This is 100% functional and will run your benchmarks perfectly fine.
