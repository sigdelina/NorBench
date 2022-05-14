from sklearn.metrics import f1_score
from eval import *


def metrics(gold, predicted):
    
    gold_lines = []
    with open(gold) as f:
        gold_lines = f.readlines()
    gold_labels = [int(line.split(' ')[1].strip()) for line in gold_lines]
    
    predincted_lines = []
    with open(predicted) as f:
        predincted_lines = f.readlines()
    pred_labels = [int(line.split(' ')[1].strip()) for line in predincted_lines]

    f1_weighted = f1_score(gold_labels, pred_labels, average='weighted')

    return f1_weighted


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--path_to_gold",
        default=None,
        type=str,
        required=True,
        help="Path to .txt file with gold labels.",
    )

    parser.add_argument(
        "--path_to_pred",
        default="",
        type=str,
        help="Path to .txt file with predicted labels.",
    )

    args = parser.parse_args()

    print('F1 weighted: ', metrics(args.path_to_gold, args.path_to_pred))

if __name__ == "__main__":
    main()



