#pragma once

#include <string>

#include "src/defs.h"

class Args {
private:
    bool is_help;

    std::string data_file;  // plik z danymi wejściowymi, obowiązkowy
    std::string test_file;  // plik z danymi testowymi, nieobowiązkowy
    std::string resp_file;  // plik z danymi na których będzie wyliczona odpowiedź
    std::string sep;        // separator w pliku: data, test, resp

    utyp    inp_col;        // ilość wejść
    utyp    out_col;        // ilość wyjść

    std::string inp_file;   // plik z którego zostanie odczytana sieć neuronowa
    std::string out_file;   // plik do którego zostanie zapiana sieć neuronowa

    ultyp    rnd_seed;      // zarodek liczb losowych

    std::string command;    // komenda do wykonania

public:
    const std::string& getDataFile() const { return data_file; }
    const std::string& getTestFile() const { return test_file; }
    const std::string& getRespFile() const { return resp_file; }
    const std::string& getSep()      const { return sep;       }

    utyp getInpCol() const { return inp_col; }
    utyp getOutCol() const { return out_col; }

    const std::string& getInpFile() const { return inp_file; }
    const std::string& getOutFile() const { return out_file; }

    ultyp getRndSeed() const { return rnd_seed; }

    const std::string& getCommand() const { return command; }

    bool isHelp() const { return is_help; }


public:
    Args();
    void help();
    void parseArgs( int argc, char *argv[] );
};


extern Args g_args;



