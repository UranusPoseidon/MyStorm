import os

class LocalFileRetriever:
    def __init__(self, directory):
        self.directory = directory

    def retrieve(self, query, top_k=10):
        results = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                with open(os.path.join(self.directory, filename), 'r') as file:
                    content = file.read()
                    if query.lower() in content.lower():
                        results.append(content)
        return results[:top_k]
