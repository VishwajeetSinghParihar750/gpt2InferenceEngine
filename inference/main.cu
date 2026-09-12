#include "model.cuh"
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

    gpt.generate(input, n_tokens);
  }
}
