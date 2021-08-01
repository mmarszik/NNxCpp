#include <QVector>
#include <cmath>
#include <QTextStream>
#include <QFile>
#include <climits>

#include "src/engine/nnnet.h"
#include "src/engine/data.h"
#include "src/misc/rand.h"

using namespace NsNet;

double classify(CNNData &data, const NNNet &nn) {

    int rightClass = 0;

    for( int i=0 ; i<data.size() ; i++ ) {
        CTVFlt obuf = nn.compute( data[i].getInps() );

        int max1 = 0;
        for( int j=1 ; j<obuf.size() ; j++ ) {
            if( obuf[j] > obuf[max1] ) {
                max1 = j;
            }
        }

        int max2 = 0;
        for( int j=1 ; j<data[i].sizeOut() ; j++ ) {
            if( data[i].getOut(j) > data[i].getOut(max2) ) {
                max2 = j;
            }
        }
        rightClass += max1 == max2;

    }

    const double ratio = 100.0 * rightClass / data.size();
//    {
//        QTextStream stdo(stdout);
//        stdo << "---------------------------------------" << endl;
//        stdo << "rightClass:" << ratio << "%" << endl;
//        stdo << "---------------------------------------" << endl;
//    }

    return ratio;
}

struct NNValue {
    NNNet nn;
    double value;
    double classify;
    bool force;
};

void experiment0() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,3,0,2,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd );
    }

    FRnd rnd(1);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-12;

    QVector< NNValue > vnn;

    //Select best 16 nn
    {
        CNNData subLearn = learn.rndSelect(2000,rnd);
        for( nnityp loop=0 ; loop<20000 ; loop++ ) {
            NNValue nn;
            nn.nn.read("/home/m/Dokumenty/c/nn/nn00/nndef.txt" );
            nn.nn.randIdxW( rnd , -1 , +1 );
            nn.value = nn.nn.error( subLearn , bigPenal );
            vnn.append( nn );
            for( nnityp i = vnn.size()-1 ; i > 0 && vnn[i].value < vnn[i-1].value ; i-- ) {
                std::swap( vnn[i] , vnn[i-1] );
            }
            if( vnn.size() > 16 ) {
                vnn.resize( 16 );
            }
            stdOut << "loop=" << loop << " current=" << nn.value << " best=" << vnn[0].value << " time=" << (time(NULL)-start) << "s" << endl;
        }
    }

    for( nnityp i=0 ; i<vnn.size() ; i++ ) {
        stdOut << " i=" << i << " value=" << vnn[i].value << " test=" << classify( test , vnn[i].nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    NNNet nn;
    {
        for( nnityp i=0 ; i<vnn.size() ; i++ ) {
            vnn[i].nn.forceIdx( learn , bigPenal , 1 , true );
            vnn[i].value = vnn[i].nn.error( learn , bigPenal );
            stdOut << " i=" << i << " value=" << vnn[i].value << " test=" << classify( test , vnn[i].nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        }
        std::sort( vnn.begin(), vnn.end(), [](const NNValue &a, const NNValue &b){ return a.value < b.value;} );
        stdOut << "summary:" << endl;
        for( nnityp i=0 ; i<vnn.size() ; i++ ) {
            stdOut << " i=" << i << " value=" << vnn[i].value << " test=" << classify( test , vnn[i].nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        }
        nn = vnn[0].nn;
    }

    //learn 4 nn.
    for( nnityp loop=1 ; true ; loop++ ) {
        if( loop==6 ) {
            nn.toUniqueWeights();
        }
        nn.learnRand1( learn, bigPenal, 1200, 0, 1E-6, 1E-2, 1, rnd(), loop>3, loop<6, false );
        nn.momentum2( learn , CNNData(), 0, 5E2, 0.01, 0.1, 1E-8, 0.8, CTVFlt(), bigPenal, 5, 1, nullptr);
        stdOut << " loop=" << loop << " learn_error=" << nn.error( learn , bigPenal ) << " test=" << classify(test,nn) << "%  time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "/home/m/Dokumenty/c/nn/nn00/nndef_out.txt" );
    }

}



void experiment1() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,3,0,2,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd );
    }

    FRnd rnd(1);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-12;

    NNNet nn;
    nn.read("/home/m/Dokumenty/c/nn/nn00/nndef.txt" );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "/home/m/Dokumenty/c/nn/nn00/nndef_out.txt" );

    nn.randIdxW( rnd , -1 , +1 );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "/home/m/Dokumenty/c/nn/nn00/nndef_out.txt" );

    nn.forceIdx( learn , bigPenal , 1E6 , true );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "/home/m/Dokumenty/c/nn/nn00/nndef_out.txt" );

    nn.learnRand1( learn, bigPenal, 3600, 0, 1E-6, 1E-2, 1, rnd(), false, true, false );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "/home/m/Dokumenty/c/nn/nn00/nndef_out.txt" );

    nn.momentum2( learn , CNNData(), 0, 5E3, 0.01, 1, 1E-8, 0.8, CTVFlt(), bigPenal, 5, 1, nullptr);
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "/home/m/Dokumenty/c/nn/nn00/nndef_out.txt" );

    nn.toUniqueWeights();

    nn.learnRand1( learn, bigPenal, 3*3600, 0, 1E-6, 1E-2, 1, rnd(), true, false, false );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "/home/m/Dokumenty/c/nn/nn00/nndef_out.txt" );

    nn.momentum2( learn , CNNData(), 0, 5E3, 0.01, 1, 1E-8, 0.8, CTVFlt(), bigPenal, 5, 1, nullptr);
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "/home/m/Dokumenty/c/nn/nn00/nndef_out.txt" );


}


int main(int argc, char *argv[])
{
    experiment1();
    return 0;
}


