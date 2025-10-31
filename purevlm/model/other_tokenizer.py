import os

import json

import sentencepiece as spm
import tiktoken

def create_sentencepiece_tokenizer(model_dir):
    """
    使用sentencepiece创建tokenizer
    """
    try:
        # 查找sentencepiece模型文件
        sp_model_path = None
        for file in os.listdir(model_dir):
            if file.endswith('.model') or file.endswith('.spm'):
                sp_model_path = os.path.join(model_dir, file)
                break
        
        if not sp_model_path:
            print("未找到sentencepiece模型文件")
            return None
        
        # 加载sentencepiece模型
        sp = spm.SentencePieceProcessor()
        sp.load(sp_model_path)
        
        class SentencePieceWrapper:
            def __init__(self, sp_processor):
                self.sp = sp_processor
                self.vocab_size = sp_processor.get_piece_size()
                self.pad_token_id = sp_processor.pad_id()
                self.eos_token_id = sp_processor.eos_id()
                self.bos_token_id = sp_processor.bos_id()
                self.unk_token_id = sp_processor.unk_id()
            
            def encode(self, text, add_special_tokens=True):
                return self.sp.encode_as_ids(text)
            
            def decode(self, token_ids, skip_special_tokens=True):
                if hasattr(token_ids, 'tolist'):
                    token_ids = token_ids.tolist()
                return self.sp.decode_ids(token_ids)
            
            def batch_decode(self, sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                results = []
                for seq in sequences:
                    results.append(self.decode(seq, skip_special_tokens))
                return results
            
            def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
                tokens = self.encode(text)
                if return_tensors == "pt":
                    import torch
                    return {"input_ids": torch.tensor([tokens])}
                return {"input_ids": tokens}
        
        return SentencePieceWrapper(sp)
        
    except Exception as e:
        print(f"创建sentencepiece tokenizer失败: {e}")
        return None


def create_tiktoken_tokenizer(model_dir):
    """
    使用tiktoken创建tokenizer
    """
    try:
        # 尝试从模型目录加载tokenizer配置
        tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 获取模型名称或编码名称
            model_name = config.get('model_name', 'gpt-4')
            
        # 创建tiktoken编码器
        encoding = tiktoken.encoding_for_model("gpt-4")  # 或者使用 "cl100k_base"
        
        class TikTokenWrapper:
            def __init__(self, encoding):
                self.encoding = encoding
                self.vocab_size = encoding.n_vocab
                self.pad_token_id = 0
                self.eos_token_id = encoding.encode("<|endoftext|>")[0] if encoding.encode("<|endoftext|>") else 0
                self.bos_token_id = self.eos_token_id
                
            def encode(self, text, add_special_tokens=True):
                return self.encoding.encode(text)
            
            def decode(self, token_ids, skip_special_tokens=True):
                if isinstance(token_ids, list):
                    return self.encoding.decode(token_ids)
                return self.encoding.decode(token_ids.tolist())
            
            def batch_decode(self, sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                results = []
                for seq in sequences:
                    if hasattr(seq, 'tolist'):
                        seq = seq.tolist()
                    results.append(self.encoding.decode(seq))
                return results
            
            def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
                tokens = self.encode(text)
                if return_tensors == "pt":
                    import torch
                    return {"input_ids": torch.tensor([tokens])}
                return {"input_ids": tokens}
        
        return TikTokenWrapper(encoding)
        
    except Exception as e:
        print(f"创建tiktoken tokenizer失败: {e}")
        return None