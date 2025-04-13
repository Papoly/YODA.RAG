import os
import re
from git import Repo
import gradio as gr
from typing import List, Dict, Tuple
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

class Agent:
    def __init__(self, system_prompt: str, llm_endpoint: str):
        self.system_prompt = system_prompt
        self.llm_endpoint = llm_endpoint
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Минимальная реализация генерации ответа"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = requests.post(
            f"{self.llm_endpoint}/chat/completions",
            json={
                "model": "local-model",
                "messages": messages,
                **kwargs
            }
        ).json()
        
        return response['choices'][0]['message']['content']

class VectorStore:
    def __init__(self):
        # Инициализация модели для эмбеддингов
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = {}
        self.documents = []
        
    def add_document(self, path: str, content: str):
        """Добавление документа в хранилище"""
        doc_id = hashlib.md5(path.encode()).hexdigest()
        self.documents.append({
            "id": doc_id,
            "path": path,
            "content": content
        })
        
        # Генерация эмбеддингов
        doc_embedding = self.model.encode(content, convert_to_tensor=False)
        self.embeddings[doc_id] = doc_embedding
        
        # Добавляем эмбеддинги для чанков
        chunks = self._chunk_content(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_embedding = self.model.encode(chunk, convert_to_tensor=False)
            self.embeddings[chunk_id] = chunk_embedding
            self.documents.append({
                "id": chunk_id,
                "path": f"{path} [chunk {i+1}]",
                "content": chunk
            })
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск наиболее релевантных документов"""
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        similarities = []
        
        for doc_id, doc_embedding in self.embeddings.items():
            sim = cosine_similarity(
                [query_embedding],
                [doc_embedding]
            )[0][0]
            similarities.append((doc_id, sim))
        
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем top_k результатов
        results = []
        for doc_id, sim in similarities[:top_k]:
            doc = next(d for d in self.documents if d["id"] == doc_id)
            results.append({
                "path": doc["path"],
                "content": doc["content"],
                "score": sim
            })
        
        return results
    
    def _chunk_content(self, content: str, max_length: int = 1000) -> List[str]:
        """Разбивка контента на чанки"""
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < max_length:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class GitRAGSystem:
    def __init__(self):
        # Инициализация компонентов (без Langfuse)
        self.llm_endpoint = "http://127.0.0.1:1234/v1"
        self.temp_dir = "./temp_repos"
        self.vector_store = VectorStore()
        
        # Создание агентов
        self.analyzer_agent = Agent(
            system_prompt="Ты - эксперт по анализу кода. Тщательно анализируй файлы из репозитория.",
            llm_endpoint=self.llm_endpoint
        )
        
        self.qa_agent = Agent(
            system_prompt="""
            Ты - помощник по работе с кодом. Отвечай на вопросы, используя предоставленную информацию.
            Всегда указывай точные источники в формате:
            Файл: [путь_к_файлу]
            Контекст: [цитата из кода]
            """,
            llm_endpoint=self.llm_endpoint
        )
        
        # Гиперпараметры по умолчанию
        self.default_generation_params = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9,
            "top_k": 40
        }
        
        # Поддерживаемые расширения файлов
        self.supported_extensions = {
            '.py', '.md', '.txt', '.js', '.jsx', '.ts', '.tsx',
            '.java', '.kt', '.scala', '.c', '.cpp', '.h', '.hpp',
            '.go', '.rs', '.rb', '.php', '.sh', '.yaml', '.yml',
            '.json', '.html', '.css', '.scss', '.sql', '.dockerfile'
        }
    
    def is_text_file(self, path: str) -> bool:
        """Проверка, является ли файл текстовым и поддерживаемым"""
        return any(path.endswith(ext) for ext in self.supported_extensions)
    
    def clone_repository(self, repo_url: str) -> str:
        """Клонирование репозитория"""
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        local_path = os.path.join(self.temp_dir, repo_name)
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        if os.path.exists(local_path):
            repo = Repo(local_path)
            repo.remotes.origin.pull()
        else:
            Repo.clone_from(repo_url, local_path)
            
        return local_path
    
    def analyze_repository(self, repo_path: str) -> List[Dict]:
        """Анализ структуры репозитория"""
        repo = Repo(repo_path)
        analysis = []
        
        for item in repo.tree().traverse():
            if item.type == 'blob' and self.is_text_file(item.path):
                try:
                    content = item.data_stream.read().decode('utf-8')
                    doc = {
                        "path": item.path,
                        "content": content,
                        "sha": item.hexsha
                    }
                    analysis.append(doc)
                    self.vector_store.add_document(item.path, content)
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Ошибка при обработке файла {item.path}: {str(e)}")
                    
        return analysis
    
    def get_relevant_context(self, question: str, top_k: int = 3) -> List[Dict]:
        """Получение релевантного контекста"""
        relevant_docs = self.vector_store.search(question, top_k=top_k)
        return [
            {
                "path": doc["path"],
                "content": doc["content"],
                "score": doc["score"]
            }
            for doc in relevant_docs
        ]
    
    def generate_answer(self, question: str, context: List[Dict], generation_params: Dict) -> Tuple[str, List[str]]:
        """Генерация ответа без Langfuse"""
        context_str = "\n\n".join([
            f"Файл: {item['path']}\nКонтент:\n{item['content']}\n(Релевантность: {item['score']:.2f})"
            for item in context
        ])
        
        prompt = f"""
        Вопрос: {question}
        
        Релевантный контекст из кодовой базы:
        {context_str}
        
        Ответь на вопрос, используя информацию из контекста. Будь точным и технически корректным.
        Всегда указывай конкретные файлы и цитируй соответствующие части кода.
        """
        
        answer = self.qa_agent.generate(prompt, **generation_params)
        
        # Извлекаем источники из ответа
        sources = []
        for line in answer.split("\n"):
            if line.startswith("Файл:"):
                sources.append(line.split("Файл:")[1].strip())
        
        return answer, sources
    
    def process_question(self, repo_url: str, question: str, temperature: float, max_tokens: int, top_p: float, top_k: int):
        """Основной процесс обработки вопроса"""
        try:
            # Клонируем и анализируем репозиторий
            repo_path = self.clone_repository(repo_url)
            self.analyze_repository(repo_path)
            
            # Получаем релевантный контекст
            context = self.get_relevant_context(question, top_k=top_k)
            
            # Генерируем ответ
            generation_params = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            answer, sources = self.generate_answer(question, context, generation_params)
            
            # Форматируем ответ для Gradio
            formatted_answer = f"{answer}\n\n**Источники:**\n" + "\n".join([f"- {src}" for src in sources])
            
            return formatted_answer
            
        except Exception as e:
            return f"Ошибка: {str(e)}"

# Инициализация системы
system = GitRAGSystem()

# Создание Gradio интерфейса
with gr.Blocks(title="Git RAG System", theme="soft") as demo:
    gr.Markdown("YODA.RAG")
    
    with gr.Row():
        with gr.Column():
            repo_url = gr.Textbox(label="URL публичного репозитория", 
                                placeholder="https://github.com/user/repo")
            question = gr.Textbox(label="Ваш вопрос", 
                                placeholder="Как работает этот проект? Опиши архитектуру.")
            submit_btn = gr.Button("Отправить", variant="primary")
        
        with gr.Column():
            output = gr.Markdown()
    
    with gr.Accordion("⚙️ Настройки генерации", open=False):
        with gr.Row():
            temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top P")
        with gr.Row():
            max_tokens = gr.Slider(100, 4000, value=2000, step=100, label="Max Tokens")
            top_k = gr.Slider(1, 10, value=3, step=1, label="Количество фрагментов контекста")
    
    gr.Examples(
        examples=[
            ["https://github.com/pytorch/pytorch", "Где реализованы основные операции тензоров?"],
            ["https://github.com/vuejs/vue", "Как работает реактивность в Vue?"]
        ],
        inputs=[repo_url, question],
        label="Примеры запросов"
    )
    
    submit_btn.click(
        fn=system.process_question,
        inputs=[repo_url, question, temperature, max_tokens, top_p, top_k],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)