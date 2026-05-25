# Doc_RAG_Prod_project
## 📦 Data Version Control (DVC) Setup & Workflow

This project uses **Git** to track code and **DVC** to manage large data files and models without bloat. Heavy files are kept out of GitHub and backed up securely in remote storage.

### 1. Installation & Environment Fix
Install the core requirements using `uv`. If you hit a file-locking error on Windows with `pywin32`, bypass it by forcing standard pip to download it first:
```powershell
# Fallback if uv hits a permission lock on pywin32
.\venv\Scripts\pip install pywin32==311

# Install the remaining project dependencies
uv pip install -r .\requirements.txt --link-mode=copy
```

### 2. Isolated Branch Workflow
Always configure and test infrastructure upgrades (like DVC) on a dedicated feature branch to keep the `main` branch clean and deployable.
```powershell
# Create and switch to a new development branch
git checkout -b feature/dvc-setup
```

### 3. Transitioning Data Tracking from Git to DVC
If Git is already tracking your data directories, you must untrack them first before DVC can safely assume management.
```powershell
# 1. Stop Git from tracking the data folder (leaves local files safe)
git rm -r --cached data/
git commit -m "Stop tracking data folder in Git to hand over to DVC"

# 2. Initialize DVC in the project
dvc init

# 3. Track the folder with DVC
dvc add data/

# 4. Save the new DVC tracking file and updated .gitignore to Git
git add data.dvc .gitignore
git commit -m "Track data folder with DVC"
```

### 4. Remote Storage Configuration
DVC requires a designated storage point to host the actual heavy raw files. 

```powershell
# Option A: Local Storage (Testing)
dvc remote add -d myremote E:\dvc_remote_storage

# Option B: Google Drive (Cloud)
dvc remote add -d myremote gdrive://YOUR_GOOGLE_DRIVE_FOLDER_ID

# Commit the configuration pointer changes
git add .dvc/config
git commit -m "Configure DVC remote storage endpoint"
```

### 5. Pushing Code and Data
To complete a milestone, push your lightweight pointer files and code modifications to GitHub, and your heavy physical files to your remote DVC storage.
```powershell
# Push code and .dvc pointer metrics to GitHub
git push origin feature/dvc-setup

# Upload actual data/model binaries to DVC storage
dvc push
```

### 6. Pulling Data (For fresh clones or collaborators)
When cloning this repository on a new machine or switching branches, fetch the matched data files instantly using:
```powershell
git pull
dvc pull
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🔄 Data Versioning & Conflict Resolution (DVC Part 2)

### 1. Tracking Folder Modifications (e.g., Deleting a File)
DVC tracks directory contents seamlessly using unique MD5 hash pointers. If you delete a tracked asset (like a raw PDF file) from your local data directories:

```powershell
# 1. Physically delete or alter the file
Remove-Item .\data\document2.pdf

# 2. Check changes (Git will show nothing; DVC will spot the modification)
dvc status

# 3. Re-index the directory state with DVC
dvc add data/

# 4. Commit the newly updated data.dvc hash to Git
git add data.dvc
git commit -m "Remove document2.pdf from raw training data"
```

### 2. Time Travel: Restoring Deleted Data
Because DVC saves full snapshots of folder histories tied to Git commits, you can instantly recover a deleted file by checking out an older Git pointer:

```powershell
# 1. Roll back the lightweight data.dvc pointer file to the previous commit
git checkout HEAD~1 data.dvc

# 2. Force DVC to reconstruct the physical files to match that old pointer
dvc checkout
```

### 3. Handling Push Rejections (`fetch first` Error)
If your `git push` fails because GitHub contains remote changes (e.g., readme edits or commits made directly on the web browser) that are missing locally, use this resolution loop:

```powershell
# 1. Fetch and merge the missing upstream commits into your local workspace
git pull origin feature/dvc-setup

# 2. Resolve any manual merge conflicts if prompted by your IDE, then:
git push origin feature/dvc-setup

# 3. Ensure your actual heavy asset binaries are synced alongside the new commits
dvc push
```

