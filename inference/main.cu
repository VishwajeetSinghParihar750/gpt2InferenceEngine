#include "classes/gpt.cuh"
#include <iostream>
#include <string>

int main() {

  Gpt2 gpt;

  while (true) {

    std::string input;
    std::cout << "\nEnter text: ";
    std::getline(std::cin, input);
    std::cout << "You entered: " << input << std::endl;

    gpt.generate(input, 20);
  }
}
