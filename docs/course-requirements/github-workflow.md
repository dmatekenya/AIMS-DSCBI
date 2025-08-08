# Student GitHub Workflow Instructions

## Overview
This document provides step-by-step instructions for working with course materials using Git and GitHub. You'll learn essential version control skills while accessing course notebooks and submitting assignments.

## Prerequisites
- A GitHub account (create one at [github.com](https://github.com) if you don't have one)
- Git and GitHub Desktop installed on your computer
- Basic familiarity with command line/terminal

## Initial Setup

### Step 1: Fork the Course Repository
1. Navigate to the [course repository](https://github.com/dmatekenya/AIMS-DSCBI)
2. Click the **"Fork"** button in the top-right corner
3. Select your GitHub account as the destination
4. Wait for GitHub to create your personal copy of the repository

### Step 2: Clone Your Fork Locally
1. On your forked repository page, click the green **"Code"** button
2. Copy the HTTPS URL it should look like: `https://github.com/dmatekenya/AIMS-DSCBI.git`
3. Open your terminal/command prompt
4. Navigate to where you want to store the course files
5. Run the clone command:
   ```bash
   git clone https://github.com/dmatekenya/AIMS-DSCBI
   ```
6. Alternatively, you can closne with GitHub Desktop
7. Navigate into the repository folder:
   ```bash
   cd COURSE-REPO-NAME
   ```
8. Open the folder in VS Code if you need to.

### Step 3: Add Upstream Remote
This allows you to get updates from the instructor's original repository:
```bash
git remote add upstream https://github.com/dmatekenya/AIMS-DSCBI.git
```

Verify your remotes:
```bash
git remote -v
```
You should see both `origin` (your fork) and `upstream` (instructor's repo).

## Getting Course Updates

When the instructor adds new materials (notebooks, assignments, etc.), follow these steps:

### Step 1: Fetch Updates from Instructor
```bash
git fetch upstream
```

### Step 2: Switch to Main Branch
```bash
git checkout main
```

### Step 3: Merge Updates
```bash
git merge upstream/main
```

### Step 4: Push Updates to Your Fork
```bash
git push origin main
```

## Working with Course Materials

### Making Changes to Notebooks
1. **Work directly on notebooks** in the appropriate folders with VS Code
2. **Save your work regularly** in Jupyter or your preferred editor
3. **Commit your changes frequently**:
   ```bash
   git add .
   git commit -m "Updated notebook exercises for week 2"
   ```
4. **Push your work to your fork**:
   ```bash
   git push origin main
   ```

*Note: Specific assignment instructions and submission procedures will be provided separately for each assignment.*







