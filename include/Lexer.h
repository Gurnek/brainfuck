#ifndef BF_LEXER_H
#define BF_LEXER_H

#include "llvm/ADT/StringRef.h"
#include <ostream>
#include <string>
#include <vector>

namespace bf {

enum Token : int {
    shift_right = '>',
    shift_left = '<',
    increment = '+',
    decrement = '-',
    input = ',',
    output = '.',
    jumpz = '[',
    jumpnz = ']'
};

std::ostream& operator<<(std::ostream& out, const Token value) {
    return out << [value]{
    #define PROCESS_VAL(p) case(p): return #p;
        switch (value) {
            PROCESS_VAL(shift_right);
            PROCESS_VAL(shift_left);
            PROCESS_VAL(increment);
            PROCESS_VAL(decrement);
            PROCESS_VAL(input);
            PROCESS_VAL(output);
            PROCESS_VAL(jumpz);
            PROCESS_VAL(jumpnz);
        }
    #undef PROCESS_VAL
    }();
}

std::vector<Token> lex_program(std::string program) {
    std::vector<Token> v_tok;
    for (char& c : program) {
        switch (c) {
            case '>':
            case '<':
            case '+':
            case '-':
            case ',':
            case '.':
            case '[':
            case ']':
                v_tok.push_back(Token(c));
                break;
            default:
                continue;
        }
    }
    return v_tok;
}

} // namespace bf
#endif // BF_LEXER_H
