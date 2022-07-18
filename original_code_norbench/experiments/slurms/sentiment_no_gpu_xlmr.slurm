#!/bin/bash
#SBATCH --job-name=xlmr_sentiment
#SBATCH --job-name=xlmr_sentiment
#SBATCH --account=nn9851k
#SBATCH --mail-type=FAIL
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=128G



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
python3 sentiment_finetuning.py --model_name "xlm-roberta" --short_model_name "xlm-roberta-base" --training_language "no" --epochs 10