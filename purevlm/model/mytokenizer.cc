#include <tokenizers_cpp.h>
#include <iostream>
#include <vector>

class TokenizersCppWrapper {
private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    
public:
    TokenizersCppWrapper(const std::string& tokenizer_path) {
        // 从文件加载tokenizer
        tokenizer = tokenizers::Tokenizer::FromFile(tokenizer_path);
        if (!tokenizer) {
            throw std::runtime_error("Failed to load tokenizer from: " + tokenizer_path);
        }
    }
    
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true) {
        auto encoding = tokenizer->Encode(text, add_special_tokens);
        return encoding.GetIds();
    }
    
    std::string decode(const std::vector<int>& token_ids, bool skip_special_tokens = true) {
        return tokenizer->Decode(token_ids, skip_special_tokens);
    }
    
    std::vector<std::string> batch_decode(const std::vector<std::vector<int>>& sequences, 
                                         bool skip_special_tokens = true) {
        std::vector<std::string> results;
        for (const auto& seq : sequences) {
            results.push_back(decode(seq, skip_special_tokens));
        }
        return results;
    }
    
    size_t vocab_size() const {
        return tokenizer->GetVocabSize();
    }
    
    int pad_token_id() const {
        auto token = tokenizer->TokenToId("<|pad|>");
        return token ? *token : 0;
    }
    
    int eos_token_id() const {
        auto token = tokenizer->TokenToId("<|endoftext|>");
        return token ? *token : 1;
    }
};

// 使用示例
int main() {
    try {
        TokenizersCppWrapper tokenizer("/path/to/tokenizer.json");
        
        std::string text = "Hello, world!";
        auto tokens = tokenizer.encode(text);
        
        std::cout << "Encoded tokens: ";
        for (int token : tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        
        std::string decoded = tokenizer.decode(tokens);
        std::cout << "Decoded text: " << decoded << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}