#!/bin/bash
#SBATCH --job-name=pos_nob_nbbert
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=15:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4



source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-scipy-ecosystem/2021.01-gomkl-2019b-Python-3.7.4
module load nlpl-transformers/4.5.1-gomkl-2019b-Python-3.7.4
module load nlpl-nlptools/2021.01-gomkl-2019b-Python-3.7.4
module load nlpl-datasets/1.17-gomkl-2019b-Python-3.7.4
module load sentencepiece/0.1.96-gomkl-2019b-Python-3.7.4

# print information (optional)
echo "submission directory: ${SUBMITDIR}"
#ulimit -a
#module list
module load nlpl-datasets/1.17-gomkl-2019b-Python-3.7.4

# by default, pass on any remaining command-line options
python3 pos_finetuning.py --model_name "nb-bert-base" --short_model_name "nb-bert-base" --training_language "nob" --epochs 10
