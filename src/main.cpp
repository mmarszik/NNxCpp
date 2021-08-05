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
        for( nnityp loop=1 ; loop<=5000 ; loop++ ) {
            NNNet nnTmp;
            nnTmp.read("nndef.txt" );
            nnTmp.randIdxW(rnd,-1.0,+1.0);
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
        CTVInt parts = {2000};
        nn.subForceIdx( learn , bigPenal , parts , 600 , 3 , rnd(), true );
        nn.save("nndef_out.txt" );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    {
        nn.learnRndIdx( learn , 600 , 60 , 1E-6 , bigPenal , rnd() , 0.40 );
        nn.save("nndef_out.txt" );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    {
        nn.learnRand1( learn, bigPenal, 600, 0, 1E-6, 1E-2, 1, rnd(), true, true, false );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save("nndef_out.txt" );
    }

    {
        nn.momentum2( learn , CNNData(), 1800, 0, 0.2, 1, 1E-9, 0.9, CTVFlt(), bigPenal, 5, nullptr);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    nn.toUniqueWeights();

    for( nnityp loop=1 ; true ; loop++ ) {
        nn.learnRand1( learn, bigPenal, 300, 0, 1E-6, 1E-2, 1, rnd(), true, false, false );
        stdOut << "loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save("nndef_out.txt" );

        nn.momentum2( learn , CNNData(), 1800, 0, 0.5, 1, 1E-12, 0.95, CTVFlt(), bigPenal, 5, nullptr);
        stdOut << "loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

}

//3,5,5; seed=1 - loop=19 learn error=0.684927 test=59.78%  time=38596s
//3,6,5: seed=1 - loop= 3 learn error=0.676044 test=62.45%  time=4934s
//3,7,5; seed=1 - loop= 1 learn error=0.67324  test=64.23%  time=2653s
//3,8,5; seed=1 - loop= 2 learn error=0.666173 test=64.895% time=4678s
//3,9,5; seed=1 - loop=16 learn error=0.662708 test=66.07% time=33074s

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
    nnftyp bigPenal = 1E-9;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    {
        CNNData subLearn = learn.rndSelect( 1000 , rnd );
        nnftyp v = nn.error( learn, bigPenal );
        for( nnityp loop=1 ; loop<=20000 ; loop++ ) {
            NNNet nnTmp;
            nnTmp.read("nndef.txt" );
            nnTmp.randIdxW(rnd);
            nncftyp vTmp = nnTmp.error( subLearn, bigPenal );
            if( vTmp < v ) {
                v  = vTmp;
                nn = nnTmp;
            }
            stdOut << "loop=" << loop << " current=" << vTmp << " best=" << v << " time=" << (time(NULL)-start) << "s" << endl;
        }
    }

    nn.save("nndef_out.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    {
        CTVInt parts = {1000};
        nn.subForceIdx( learn , bigPenal , parts , 300 , 3 , rnd(), true );
        nn.save("nndef_out.txt" );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    {
        nn.forceIdx( learn , bigPenal , 1E3 , true );
        nn.save("nndef_out.txt" );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    {
        nn.learnRndIdx( learn , 300 , 0 , 1E-6 , bigPenal , rnd() , 0.5 );
        nn.save("nndef_out.txt" );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    {
        nn.momentum2( learn, CNNData(), 1800, 0, 0.1, 1, 1E-9, 0.95, CTVFlt(), bigPenal, 5 );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    nn.toUniqueWeights();

    {
        NNNet bestTest = nn;
        CNNData subLearn = learn.rndSelect(8000,rnd);
        CNNData subTest  = learn.rndSelect(8000,rnd);
        nn.momentum2( subLearn, subTest, 2400, 20, 0.05, 1, 1E-9, 0.995, CTVFlt(), bigPenal, 5, &bestTest);
        nn = bestTest;
        stdOut << " momentum2 with test" << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    for( nnityp loop=1 ; true ; loop++ ) {
        nn.learnRand1( learn, bigPenal, 300, 0, 1E-12, 1E-3, loop==1 ? 1 : 0.1, rnd(), true, false, false );
        stdOut << " learnRand1; loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save("nndef_out.txt" );

        nn.momentum2( learn , CNNData(), 1800, 0, 0.05, 1, 1E-12, 0.995, CTVFlt(), bigPenal, 5, nullptr);
        stdOut << "momentum2; loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

}


int main(int argc, char *argv[])
{
    experiment1();
    return 0;
}


