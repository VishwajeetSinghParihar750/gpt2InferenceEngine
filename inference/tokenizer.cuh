#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "constants.hh"
#include "include/json.hpp"

class Gpt2Tokenizer {
  std::vector<std::string> vocab;
  std::unordered_map<std::string, int> gpt2TokenToTokenId;
  std::map<std::pair<std::string, std::string>, int> merges;

  int GetTokenIdFromToken(std::string token) {
    auto it = gpt2TokenToTokenId.find(token);
    assert(it != gpt2TokenToTokenId.end());
    return it->second;
  }

  void ParseMerges(const nlohmann::json &mergesJson) {
    this->merges.clear();
    int priority = 0;
    for (const auto &merge : mergesJson) {
      this->merges[{merge[0].get<std::string>(), merge[1].get<std::string>()}] =
          priority++;
    }
  }

public:
  Gpt2Tokenizer() {
    std::cout << "loading vocab ... " << std::endl;

    using json = nlohmann::json;

    std::ifstream f("../weights/tokenizer/tokenizer.json");
    assert(f.is_open() && "failed to open tokenizer.json");

    const json data = json::parse(f);

    this->vocab.assign(VOCAB_SIZE, "");
    gpt2TokenToTokenId.clear();

    for (const auto &[token, tokenIdJson] : data["model"]["vocab"].items()) {
      const int tokenId = tokenIdJson.get<int>();
      assert(tokenId >= 0 && tokenId < VOCAB_SIZE);
      this->vocab[tokenId] = token;
      gpt2TokenToTokenId[token] = tokenId;
    }

    if (data.contains("added_tokens")) {
      for (const auto &added : data["added_tokens"]) {
        const int tokenId = added["id"].get<int>();
        const std::string content = added["content"].get<std::string>();
        if (tokenId >= static_cast<int>(this->vocab.size()))
          this->vocab.resize(tokenId + 1);
        this->vocab[tokenId] = content;
        gpt2TokenToTokenId[content] = tokenId;
      }
    }

    ParseMerges(data["model"]["merges"]);
    std::cout << "vocab loaded..." << std::endl;
  }

  std::vector<int> encode(std::string inputStr) {
    // regex based chunks, then chunks split into chars
    std::regex rg(" ?[A-Za-z]+| ?[0-9]+| ?[^ A-Za-z0-9]+|\\s+");
    std::sregex_iterator it(inputStr.begin(), inputStr.end(), rg);
    std::sregex_iterator end;

    std::vector<std::vector<std::string>> preChunksSplit;

    for (; it != end; it++) {
      std::smatch match = *it;
      std::string chunkStr = match.str();

      preChunksSplit.emplace_back();

      size_t startIdx = 0;
      if (!chunkStr.empty() && chunkStr[0] == ' ') {
        preChunksSplit.back().push_back("\u0120"); // Ġ, as one unit
        startIdx = 1;
      }
      for (size_t i = startIdx; i < chunkStr.size(); i++) {
        preChunksSplit.back().push_back(std::string(1, chunkStr[i]));
      }
    }

    // run bpe per chunk
    for (auto &chunk : preChunksSplit) {
      while (true) {
        int len = chunk.size();
        int mergePosition = -1;
        int mergePriority = 1e9;
        for (int i = 0; i < len - 1; i++) {
          int priority = 1e9;
          if (this->merges.contains(std::make_pair(chunk[i], chunk[i + 1]))) {
            priority = this->merges[std::make_pair(chunk[i], chunk[i + 1])];
          }

          if (priority < mergePriority) {
            mergePriority = priority;
            mergePosition = i;
          }
        }

        if (mergePosition == -1)
          break;
        else {
          chunk[mergePosition] += chunk[mergePosition + 1];
          chunk.erase(chunk.begin() + mergePosition + 1);
        }
      }
    }

    std::vector<int> result;

    for (const auto &i : preChunksSplit) {
      for (const auto &token : i) {
        result.push_back(GetTokenIdFromToken(token));
      }
    }

    return result;
  }

  std::string decode(int id) {
    assert(id >= 0 && id < static_cast<int>(vocab.size()));
    const std::string &token = vocab[id];

    std::string decoded;
    decoded.reserve(token.size());
    const std::string gpt2Space = "\u0120";   // Ġ
    const std::string gpt2Newline = "\u010A"; // Ċ
    for (size_t i = 0; i < token.size();) {
      if (token.compare(i, gpt2Space.size(), gpt2Space) == 0) {
        decoded += ' ';
        i += gpt2Space.size();
      } else if (token.compare(i, gpt2Newline.size(), gpt2Newline) == 0) {
        decoded += '\n';
        i += gpt2Newline.size();
      } else {
        decoded += token[i++];
      }
    }
    return decoded;
  }
};
