from vectorsearch import VectorSearch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--vtt", type=str, required=True, help="Путь к субтитрам vtt")
parser.add_argument("--idx", type=str, default="index", help="Название индекса для поиска")
args = parser.parse_args()

vs = VectorSearch()
vs.add_vtt_file(args.vtt)
vs.build_index()

vs.save_index(args.idx)
print(f"Индекс сохранен в {args.idx}.ann и документы в {args.idx}.pkl")