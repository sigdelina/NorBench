from utils import *


def _create_records_from_csv(csv_path, deserialize_fn):
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # read off header

        return [deserialize_fn(row) for row in reader]


def write_golds(output_dir, csv_path, ids, predictions):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_name = os.path.basename(csv_path).split('.')[0]
    output_file = os.path.join(
        output_dir,
        f"{dataset_name}_golds.txt"
    )
    
    with open(output_file, "w") as f:
        for ind, gold in zip(ids, predictions):
            print(f"{ind} {gold}", file=f)


def create_golds(path_to_csv, output_dir):
    dataframe = pd.read_csv(path_to_csv)
    ids = dataframe.id.to_list()
    gold_keys = [eval(row[2])[[int(t) for t in eval(row[3])][0]] for ind, row in dataframe.iterrows()]
    write_golds(output_dir, path_to_csv, ids, gold_keys)
    


def main():
    path_to_csv = "wsd_bert/data/corpus-max_num_gloss=4.csv"
    output_dir = "wsd_bert/data/"
    create_golds(path_to_csv, output_dir)

if __name__ == "__main__":
    main()