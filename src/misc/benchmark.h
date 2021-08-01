#pragma once
/*
#include <QElapsedTimer>
#include <QString>
#include <QTextStream>


class Benchmark {
private:
    unsigned int  count;
    QElapsedTimer timer;
    qint64        elapsed;

private:

public:
    Benchmark() : count(0), elapsed(0) {}

    void reset() { count=0; elapsed=0; }
    void start() { count++; timer.start(); }
    void end()   { elapsed += timer.elapsed(); }

    int  getCount()     const { return count;     }
    double getElapsed() const { return elapsed/1000.0; }

    QString getRaport() const { return QString( "time: %1s; count=%2" ).arg( getElapsed() , 12 , 'f' , 6 , ' ' ).arg( count ); }

    void show()  { QTextStream out(stdout); out << getRaport() << endl; }

};



extern Benchmark ben1;
extern Benchmark ben2;
*/


#include <chrono>
#include <string>
#include <iostream>
#include <sstream>

class Benchmark {
private:
    unsigned int  count;
    std::chrono::steady_clock::time_point timer;
    unsigned long long elapsed;

private:

public:
    Benchmark() : count(0), elapsed(0) {}

    void reset() { count=0; elapsed=0; }
    void start() { count++; timer = std::chrono::steady_clock::now();  }
    void end()   { elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-timer).count() ;  }

    int  getCount()     const { return count;     }
    double getElapsed() const { return elapsed/1000000000.0; }

    std::string getRaport() const {
        std::stringstream ss;
        ss.precision(9);
        ss << std::fixed;
        ss << "time: " << getElapsed() << "s; count: " << getCount();
        return ss.str();
    }
    void show()  {
        std::cout << getRaport() << std::endl;
    }

};



extern Benchmark ben1;
extern Benchmark ben2;
extern Benchmark ben3;
extern Benchmark ben4;
