# SonarQube Analysis

This directory contains the configuration and scripts for running automated SonarQube analysis on cuOpt branches.

## Files

- `sonar-branches.txt` - List of branches to analyze (one per line)
- `run-sonar-analysis.sh` - Automated script that clones, builds, and analyzes branches

## Quick Start

### 1. Configure Branches

Edit `sonar-branches.txt` to specify which branches to analyze:

```bash
# One branch per line
main
release/26.02

# Lines starting with # are comments
# Empty lines are ignored
```

### 2. Set Required Environment Variable

The script requires authentication:

```bash
export SONAR_TOKEN="your_token_here"
```

**Note**: Contact the cuOpt team for token details.

### 3. Run the Analysis

```bash
cd /path/to/cuopt
./sonarqube/run-sonar-analysis.sh
```

## Script Behavior

The script will automatically:

1. ✅ Validate branch configuration file exists and has at least one branch
2. ✅ Clone each branch into a fresh temporary directory
3. ✅ Create an isolated conda environment per branch
4. ✅ Build the project using `./build.sh`
5. ✅ Run SonarQube analysis with branch-specific tagging
6. ✅ Clean up temporary files and conda environments
7. ✅ Provide a summary of successful and failed branches

## Support

**Contact**: cuOpt team

For issues with:
- Build failures: See [CONTRIBUTING.md](../CONTRIBUTING.md)
- Script bugs: Report to the cuOpt team
