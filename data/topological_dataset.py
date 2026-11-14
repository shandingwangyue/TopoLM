import torch
from torch.utils.data import Dataset
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from datasets import load_dataset
import re
import os

# ==========================================
# 修复 NLTK 在代理网络下的下载安全拦截
# ==========================================
import nltk.pathsec
os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
nltk.pathsec.ALLOW_PROXIED_FETCH = True
# ==========================================


class RealConceptVocab:
    def __init__(self):
        self.word2id = {"": 0, "": 1}
        self.id2word = {0: "", 1: ""}
        self.idx = 2

    def add_concept(self, word):
        word = word.lower().strip()
        # 移除了容量限制，只要是新词就全部收入拓扑宇宙
        if word not in self.word2id:
            self.word2id[word] = self.idx
            self.id2word[self.idx] = word
            self.idx += 1
        return self.word2id[word]

    def get_id(self, word):
        return self.word2id.get(word.lower().strip(), 1)

    def __len__(self):
        return len(self.word2id)

class WordNetTopologyParser:
    @staticmethod
    def extract_relations(vocab):
        # 自动下载 WordNet 数据
        try:
            wn.all_synsets()
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

        inclusions, contradictions = [], []
        for synset in wn.all_synsets():
            lemmas = [l.name().lower() for l in synset.lemmas() if '_' not in l.name()]
            if not lemmas: continue
            
            anchor_id = vocab.add_concept(lemmas[0])
            
            # 提取正向包含关系 (Hypernyms)
            for hypernym in synset.hypernyms():
                h_lemmas = [l.name().lower() for l in hypernym.lemmas() if '_' not in l.name()]
                if h_lemmas:
                    target_id = vocab.add_concept(h_lemmas[0])
                    inclusions.append((anchor_id, target_id, 1.0))
            
            # 提取负向互斥关系 (Antonyms)
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    ant_word = antonym.name().lower()
                    if '_' not in ant_word:
                        ant_id = vocab.add_concept(ant_word)
                        contradictions.append((anchor_id, ant_id, -1.0))
                        
        print(f"[数据] 提取常识拓扑: 包含关系 {len(inclusions)} 条, 互斥关系 {len(contradictions)} 条")
        return inclusions, contradictions

class NLTKTextParser:
    @staticmethod
    def extract_sequences(vocab, seq_len=8):
        print("[数据] 正在加载 NLTK Brown 完整真实无监督语料...")
        try:
            from nltk.corpus import brown
            brown.sents()
        except LookupError:
            import nltk
            nltk.download('brown', quiet=True)
            from nltk.corpus import brown
            
        sequences = []
        
        # 遍历完语料库里的每一句话，不设任何截断
        for sentence in brown.sents():
            words = [w.lower() for w in sentence if w.isalpha()]
            if len(words) < 5: continue
            
            # 这里即使是生僻词，只要前面没出现过，也会被加入词表
            tokens = [vocab.add_concept(w) for w in words]
            for i in range(0, len(tokens) - seq_len):
                seq = tokens[i : i + seq_len]
                sequences.append(seq)
                
        print(f"[数据] 提取语言上下文: {len(sequences)} 个完整序列")
        return sequences
class TopoCurriculumDataset(Dataset):
    """支持双轨课程学习的混合数据集"""
    def __init__(self, structural_data, text_data, seq_len=8):
        self.structural_data = structural_data
        self.text_data = text_data
        self.seq_len = seq_len
        self.phase = 'rules' # 初始锁定在物理常识构建阶段

    def set_phase(self, phase_name):
        assert phase_name in ['rules', 'unsupervised']
        self.phase = phase_name

    def __len__(self):
        return len(self.structural_data) if self.phase == 'rules' else len(self.text_data)

    def __getitem__(self, idx):
        if self.phase == 'rules':
            anchor, target, relation = self.structural_data[idx]
            context = [anchor, target] + [0] * (self.seq_len - 2)
            return torch.tensor(context, dtype=torch.long), torch.tensor(relation, dtype=torch.float32)
        else:
            # ========================================================
            # 【机制创新：连续拓扑锚定 (Continuous Grounding)】
            # 在无监督文本学习中，保留 10% 的概率唤醒物理常识回忆。
            # 这相当于人类的“具身认知”：在阅读时依然保持对物理世界的感知，
            # 彻底抵御语言模型带来的灾难性遗忘与拓扑坍缩！
            # ========================================================
            if torch.rand(1).item() < 0.10 and len(self.structural_data) > 0:
                struct_idx = torch.randint(0, len(self.structural_data), (1,)).item()
                anchor, target, relation = self.structural_data[struct_idx]
                context = [anchor, target] + [0] * (self.seq_len - 2)
                return torch.tensor(context, dtype=torch.long), torch.tensor(relation, dtype=torch.float32)
            else:
                context = self.text_data[idx]
                return torch.tensor(context, dtype=torch.long), torch.tensor(0.0, dtype=torch.float32)