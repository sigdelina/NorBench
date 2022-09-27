#!/bin/bash
#SBATCH --job-name=lm_eval_sentiment
#SBATCH --account=nn9851k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=8G
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

RUN_NAME=${1}
MODEL_NAME=${2}
LANG=${3}  # no or any other subdirectory in data/sentiment/


# by default, pass on any remaining command-line options
python3 sentiment_finetuning.py --model_name ${RUN_NAME} --short_model_name ${MODEL_NAME} --training_language ${LANG} --epochs 10
