# MicroBooNE axial form factor measurement

This repository contains the PROfit XML configurations, Python scripts, and analysis notebooks used for the MicroBooNE axial form factor measurement.

## Git setup

Configure the notebook filter once before committing. It removes outputs from Git while keeping them in local working copies:

```bash
git config --global filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
git config --global filter.strip-notebook-output.required true
```
