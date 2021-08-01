#pragma once

#include <string>
#include <iostream>
#include <sstream>


class ExCrit {
private:
    const std::string msg;
    const std::string file;
    const int         line;
public:
    ExCrit( const std::string &msg , const std::string &file, const int line ) : msg(msg), file(file), line(line) {
        std::cout<<toStr();
    }
    std::string toStr() const {
        std::stringstream str;
        str << "msg:"  << msg << std::endl;
        str << "file:" << file << std::endl;
        str << "line:" << line << std::endl;
        return str.str();
    }
};


//#define  EXCP( __exp__  ) if( !( __exp__ ) ) throw ExCrit( #__exp__ , __FILE__ , __LINE__ )
#define  PEXCP( __exp__ ) if( !( __exp__ ) ) throw ExCrit( #__exp__ , __FILE__ , __LINE__ )
#define  EXCP( __exp__  )

