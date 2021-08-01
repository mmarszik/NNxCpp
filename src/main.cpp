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

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    {
        CNNData subLearn = learn.rndSelect( 1000 , rnd );
        nnftyp v = nn.error( learn, bigPenal );
        for( nnityp loop=1 ; loop<=100 ; loop++ ) {
            NNNet nnTmp;
            nnTmp.read("nndef.txt" );
            nnTmp.randIdxW(rnd,-1,+1);
            nncftyp vTmp = nnTmp.error( subLearn, bigPenal );
            if( vTmp < v ) {
                v = vTmp;
                nn = nnTmp;
            }
            stdOut << "loop=" << loop << " current=" << vTmp << " best=" << v << " time=" << (time(NULL)-start) << "s" << endl;
        }
    }

    nn.save("nndef_out.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    {
        CTVInt parts = {200,200};
        nn.subForceIdx( learn , bigPenal , parts , 600 , 3 , rnd() );
        nn.save("nndef_out.txt" );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    {
        CTVInt parts = {1000,1000};
        nn.subForceIdx( learn , bigPenal , parts , 3600 , 3 , rnd() );
        nn.save("nndef_out.txt" );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    for( nnityp loop=1 ; true ; loop++ ) {
        nn.learnRand1( learn, bigPenal, 600, 0, 1E-6, 1E-2, 1, rnd(), false, true, false );
        stdOut << "loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save("nndef_out.txt" );
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
    nn.read("nndef.txt" );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    nn.randIdxW( rnd , -1 , +1 );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    nn.forceIdx( learn , bigPenal , 1E6 , true );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    nn.learnRand1( learn, bigPenal, 3600, 0, 1E-6, 1E-2, 1, rnd(), false, true, false );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    nn.momentum2( learn , CNNData(), 0, 5E3, 0.01, 1, 1E-8, 0.8, CTVFlt(), bigPenal, 5, 1, nullptr);
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    nn.toUniqueWeights();

    nn.learnRand1( learn, bigPenal, 3*3600, 0, 1E-6, 1E-2, 1, rnd(), true, false, false );
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    nn.momentum2( learn , CNNData(), 0, 5E3, 0.01, 1, 1E-8, 0.8, CTVFlt(), bigPenal, 5, 1, nullptr);
    stdOut << " current=" << nn.error(learn,bigPenal) << " test=" << classify(test,nn) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

}


int main(int argc, char *argv[])
{
    experiment0();
    return 0;
}


