#!/bin/bash
#SBATCH --job-name=sent_nbbase
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4



source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-scipy-ecosystem/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-transformers/4.20.1-gomkl-2021a-Python-3.9.5
module load nlpl-nlptools/2022.01-gomkl-2021a-Python-3.9.5
module load nlpl-datasets/2.3.2-gomkl-2021a-Python-3.9.5
module load nlpl-sentencepiece/0.1.96-gomkl-2021a-Python-3.9.5

# print information (optional)
echo "submission directory: ${SUBMITDIR}"
#ulimit -a
#module list

# by default, pass on any remaining command-line options
python3 sentiment_finetuning.py --model_name "nb-bert-base" --short_model_name "nb-bert-base" --training_language "no" --epochs 10
