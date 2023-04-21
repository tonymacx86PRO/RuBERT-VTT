
from transformers import AutoTokenizer, AutoModel
from annoy import AnnoyIndex
import webvtt
import torch
import pickle

class VectorSearch:
    def __init__(self):
        # Загрузка токенизатора и модели RuBERT
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

        # Создание индекса Annoy
        self.index = AnnoyIndex(768, 'angular')
        self.documents = []

    def add_documents(self, documents):
        # Добавление документов в индекс
        for doc in documents:
            # Токенизация и кодирование документа
            input_ids = self.tokenizer.encode(doc, return_tensors="pt")
            with torch.no_grad():
                # Получение эмбеддинга документа
                embedding = self.model(input_ids)[0].mean(dim=1).squeeze().numpy()
            # Добавление эмбеддинга в индекс
            self.index.add_item(len(self.documents), embedding)
            self.documents.append(doc)

    def add_vtt_file(self, filename):
        # Парсинг файла vtt
        captions = webvtt.read(filename)
        # Добавление субтитров в индекс
        for caption in captions:
            self.add_documents([caption.text])

    def build_index(self):
        # Построение индекса
        self.index.build(10000)

    def search(self, query, n=2):
        # Поиск ближайших документов к запросу
        input_ids = self.tokenizer.encode(query, return_tensors="pt")
        with torch.no_grad():
            query_embedding = self.model(input_ids)[0].mean(dim=1).squeeze().numpy()
        indices = self.index.get_nns_by_vector(query_embedding, n)

        # Вывод результатов поиска
        for index in indices:
            print(self.documents[index])

    def save_index(self, filename):
        # Сохранение индекса и списка документов на диск
        self.index.save(f"{filename}.ann")
        with open(filename + ".docs", "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self, filename):
        # Загрузка индекса и списка документов с диска
        self.index.load(f"{filename}.ann")
        with open(filename + ".docs", "rb") as f:
            self.documents = pickle.load(f)