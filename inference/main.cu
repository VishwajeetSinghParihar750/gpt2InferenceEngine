#include "model.cuh"
#include <chrono>
#include <iostream>
#include <string>

int main() {

  Gpt2 gpt;

  while (true) {

    std::string input;
    std::cout << "\nEnter text: ";
    std::getline(std::cin, input);

    int n_tokens;
    std::cout << "Enter output token count: ";
    std::cin >> n_tokens;
    std::cin.ignore(); // clear leftover newline for next getline

    std::cout << "You entered: " << input << " (" << n_tokens << " tokens)"
              << std::endl;

    const auto t0 = std::chrono::steady_clock::now();
    gpt.generate(input, n_tokens);
    const auto t1 = std::chrono::steady_clock::now();

    const double secs =
        std::chrono::duration<double>(t1 - t0).count();
    const double tokPerSec = n_tokens > 0 ? n_tokens / secs : 0.0;
    std::cout << "\n[" << n_tokens << " tokens in " << secs << "s => "
              << tokPerSec << " tok/sec]" << std::endl;
  }
}
