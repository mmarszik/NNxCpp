
#include <ctime>

#include "args.h"
#include "critical.h"
#include "tools.h"


Args g_args;

Args::Args() {
    is_help = false;
    rnd_seed = time(NULL);
    sep = ",";
}

void Args::help() {
    std::cout<<"using:"<<std::endl<<"nnlc01"<<std::endl;
    std::cout<<"data params:"<<std::endl;
        std::cout<<"\t"<<"data_file=file contain learn data"<<std::endl;
        std::cout<<"\t"<<"test_file=file contain test data"<<std::endl;
        std::cout<<"\t"<<"resp_file=file contain response data"<<std::endl;
        std::cout<<"\t"<<"sep=separator in data, test and response file"<<std::endl;
        std::cout<<"\t"<<"inp_col=number of input column"<<std::endl;
        std::cout<<"\t"<<"out_col=number of output column"<<std::endl;

    std::cout<<"neural net params:"<<std::endl;
        std::cout<<"\t"<<"inp_file=file contain neural network"<<std::endl;
        std::cout<<"\t"<<"out_file=file to save neural network"<<std::endl;

    std::cout<<"random params:"<<std::endl;
        std::cout<<"\t"<<"rnd_seed=random seed"<<std::endl;

    std::cout<<"commands="<<std::endl;
        std::cout<<"\t"<<"learn0; learn average and multiregression"<<std::endl;
    abort();
}


void Args::parseArgs(int argc, char *argv[] ) {
	for( int i=1 ; i<argc ; i++ ) {
        const std::string tmp = argv[i];
        const size_t idx = tmp.find("=");
        if( idx == std::string::npos ) {
            is_help=true;
            break;
        }

        const std::string val = trim( tmp.substr(idx+1) );
        const std::string arg = trim( tmp.substr(0,idx) );

        if( false ) {
        } else if( arg == "data_file" ) {
            data_file = val;
        } else if( arg == "test_file" ) {
            test_file = val;
        } else if( arg == "resp_file" ) {
            resp_file = val;
        } else if( arg == "sep" ) {
            sep = val;
        } else if( arg == "inp_col" ) {
            try { inp_col = std::stoi( val ); } catch(...) { help(); }
        } else if( arg == "out_col" ) {
            try { out_col = std::stoi( val ); } catch(...) { help(); }
        } else if( arg == "inp_file" ) {
            inp_file = val;
        } else if( arg == "out_file" ) {
            out_file = val;
        } else if( arg == "rnd_seed" ) {
            try { rnd_seed = std::stoull( val ); } catch(...) { help(); }
        } else if( arg == "command" ) {
            command = val;
            if( command != "learn0" )
                help();
        } else {
            is_help = true;
        }
	}
}




