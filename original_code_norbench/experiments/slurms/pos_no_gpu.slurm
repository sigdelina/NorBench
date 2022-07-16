#!/bin/bash
#SBATCH --job-name=bert_wordnet
#SBATCH --job-name=in5550
#SBATCH --account=nn9851k
#SBATCH --mail-type=FAIL
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=80G

source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all
module load nlpl-scipy-ecosystem/2021.01-gomkl-2019b-Python-3.7.4
module load nlpl-transformers/4.5.1-gomkl-2019b-Python-3.7.4

# print information (optional)
echo "submission directory: ${SUBMITDIR}"
#ulimit -a
#module list

# by default, pass on any remaining command-line options
python3 pos_finetuning.py --model_name "norbert" --short_model_name "/cluster/shared/nlpl/data/vectors/latest/216/" --training_language "nno" --epochs 10