# .gitignore Changes

## Summary
Updated `.gitignore` to exclude large files and directories from version control, significantly reducing repository size.

## Added Entries

### Model Checkpoints and Experiments (~973MB)
- `exps/` - Contains model checkpoints (e.g., `checkpoint0099.pth` is 973MB)
- `*.pth` - PyTorch model files
- `*.ckpt` - Checkpoint files
- `*.weights` - Model weight files

### Dataset Files
- `KITTI/` - KITTI dataset directory
- `data/` - General data directory
- `*.jpg`, `*.jpeg`, `*.png` - Image files (should be stored separately)
- `*.mp4`, `*.avi` - Video files (should be stored separately)

### Build Artifacts (~25MB)
- `models/ops/build/` - Compiled build files
- `*.o` - Object files
- `*.so` - Shared object libraries (already in base .gitignore, but reinforced)
- `*.a` - Static libraries

### Generated Outputs
- `output/` - Output directory
- `results/` - Results directory
- `crops/` - LLM filter crop images
- `visualization/` - Visualization outputs
- `*.txt.bak` - Backup text files

### Tracking Evaluation Data
- `TrackEval/data/trackers/` - Tracker results
- `TrackEval/data/gt/` - Ground truth data

### IDE and Editor Files
- `.vscode/` - VS Code settings
- `.idea/` - JetBrains IDE settings
- `*.swp`, `*.swo`, `*~` - Vim/editor temporary files

### Temporary Files
- `*.tmp` - Temporary files
- `*.temp` - Temporary files

## Recommendations

### Before Committing
If you have already committed large files, you'll need to remove them from git history:

```bash
# Remove files from git tracking but keep them locally
git rm --cached -r exps/
git rm --cached -r models/ops/build/

# Commit the removal
git commit -m "Remove large files from tracking"
```

### For Already Committed Large Files
If large files are already in git history and you want to clean them:

```bash
# Use git filter-repo (recommended) or BFG Repo-Cleaner
# Install git-filter-repo first
pip install git-filter-repo

# Remove directory from entire history
git filter-repo --path exps --invert-paths
git filter-repo --path models/ops/build --invert-paths

# Force push (WARNING: This rewrites history)
git push origin --force --all
```

### Storage Best Practices
1. **Model checkpoints**: Store in cloud storage (Google Drive, S3, DVC)
2. **Datasets**: Use symlinks or download scripts
3. **Large results**: Archive separately or use Git LFS
4. **Build artifacts**: Always rebuild from source

## Impact
- **Excluded ~998MB** from being tracked in git
- Repository size will be much smaller after cleanup
- Faster clone/push/pull operations
- Reduced storage costs for remote repositories
