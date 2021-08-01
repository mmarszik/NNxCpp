#include "tools.h"

std::string trim( std::string s ) {
    const static char *const ws = " \t\n\r\f\v";
    s.erase( 0 , s.find_first_not_of( ws ) );
    s.erase( s.find_last_not_of( ws ) + 1 );
    return s;
}
