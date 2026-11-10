#!/bin/bash
#PBS -N Hubble
#PBS -l nodes=1:ppn=1
#PBS -l walltime=30:00:00
#PBS -o Hubble.log
#PBS -e Hubble.0.log

cd "$PBS_O_WORKDIR" || { echo "cd to PBS_O_WORKDIR failed"; exit 1; }

# --- Proper conda initialization for a non-interactive batch shell ---
# `source ~/.bashrc` often does NOT activate conda here, because most
# .bashrc files wrap the conda init block in an interactive-shell check
# (e.g. `case $- in *i*) ... ;; esac`) that a PBS job does not satisfy.
# Source conda.sh directly instead, which works regardless.
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    # fallback: try common install locations if `conda info` isn't on PATH yet
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            source "$candidate/etc/profile.d/conda.sh"
            break
        fi
    done
fi

conda activate "${HZ_CONDA_ENV:-camb_env}" || { echo "conda activate "${HZ_CONDA_ENV:-camb_env}" failed"; exit 1; }

# --- Diagnostics: print exactly what environment/interpreter is active ---
echo "which python: $(which python)"
echo "which python3: $(which python3)"
echo "python executable: $(python -c 'import sys; print(sys.executable)')"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
python -c "import sys; print('sys.path:'); [print(' ', p) for p in sys.path]"

# --- Dependency check: fail fast with a clear message, not a bare traceback ---
python -c "import torch; print('torch', torch.__version__, 'OK, from', torch.__file__)" \
    || { echo "FATAL: torch not importable from the python being used above."; \
         echo "Check: is torch installed in THIS exact env/interpreter (see paths above)?"; \
         echo "Try: conda list torch   OR   pip show torch   inside this same job/shell."; \
         exit 1; }
python -c "import sklearn, scipy, pandas, matplotlib; print('sklearn/scipy/pandas/matplotlib OK')" \
    || { echo "FATAL: one of sklearn/scipy/pandas/matplotlib missing from this env."; exit 1; }

ls -la scripts/run_all.sh || { echo "scripts/run_all.sh not found in $PBS_O_WORKDIR (submit from the repo root)"; exit 1; }

echo "Running the full analysis (scripts/run_all.sh)..."
bash scripts/run_all.sh
echo "Done."
