from vectorsearch import VectorSearch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, help="Поисковый запрос по индексу")
parser.add_argument("--idx", type=str, default="index", help="Индекс для поиска")
parser.add_argument("--n", type=int, default=2, help="Количество ближайших документов")
args = parser.parse_args()

vs = VectorSearch()
vs.load_index(args.idx)
vs.search(args.query, args.n)