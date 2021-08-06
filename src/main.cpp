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

    if( nn.error( learn, bigPenal) > 0.69 ) {
        nn.randIdxW(rnd,-1,+1);
        nn.save( "nndef_out.txt" );
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    nn.toUniqueWeights();
    nn.save( "nndef_out.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;


    TVInt sizes = {100,500,1000,2000,4000,8000,12000,16000,24000,30000};
    for( nnityp loop=0 ; loop < sizes.size() ; loop++ )
    {
        NNNet bestTest = nn;
        CNNData subLearn = learn.rndSelect( sizes[loop] , rnd );
        CNNData subTest  = learn.rndSelect( sizes[loop] , rnd );
        nn.momentum2( subLearn, subTest, 1200, 20, 0.05, 0.1, 1E-9, 0.99, CTVFlt(), bigPenal, 5, &bestTest);
        nn = bestTest;
        stdOut << " momentum2 with test; size:" << sizes[loop] << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    for( nnityp loop=1 ; true ; loop++ ) {
        nn.learnRand1( learn, bigPenal, 300, 0, 1E-12, 1E-3, loop==1 ? 1 : 0.1, rnd(), true, false, false );
        stdOut << " learnRand1; loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save("nndef_out.txt" );

        nn.momentum2( learn , CNNData(), 1500, 0, 0.05, 0.1, 1E-12, 0.99, CTVFlt(), bigPenal, 5, nullptr);
        stdOut << "momentum2; loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }


}

//3, 5,5;       seed=1 - loop=19 learn error=0.684927 test=59.78%  time=38596s
//3,10,10,10,5; seed=1 - loop=5  learn error=0.585161 test=83.02%  time=19532s



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
    nn.read( "nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    {
        CNNData subLearn = learn.rndSelect( 1000 , rnd );
        nnftyp v = nn.error( learn, bigPenal );
        for( nnityp loop=1 ; loop<=20000 ; loop++ ) {
            NNNet nnTmp;
            nnTmp.read("nndef.txt" );
            nnTmp.randIdxW( rnd , -1.0 , +1.0 );
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
        stdOut << "subForceIdx; learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    {
        bool work = true;
        for( nnityp loop=1 ; loop <= 30 && work ; loop++ )
        {
            work = nn.forceIdx( learn , bigPenal , 1 , true );
            nn.save("nndef_out.txt" );
            stdOut << "forceIdx; loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        }
    }

    {
        nn.learnRndIdx( learn , 300 , 0 , 1E-6 , bigPenal , rnd() , 0.5 );
        nn.save("nndef_out.txt" );
        stdOut << "learnRndIdx; learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    {
        nn.momentum2( learn, CNNData(), 3600, 0, 0.1, 1, 1E-9, 0.95, CTVFlt(), bigPenal, 5 );
        stdOut << "small momentum2; learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    nn.toUniqueWeights();

    TVInt sizes = {1000,2000,8000};
    for( nnityp loop=0 ; loop < sizes.size() ; loop++ )
    {
        NNNet bestTest = nn;
        CNNData subLearn = learn.rndSelect( sizes[loop] , rnd );
        CNNData subTest  = learn.rndSelect( sizes[loop] , rnd );
        nn.momentum2( subLearn, subTest, 1200, 20, 0.05, 1, 1E-9, 0.995, CTVFlt(), bigPenal, 5, &bestTest);
        nn = bestTest;
        stdOut << " momentum2 with test; size:" << sizes[loop] << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
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
    experiment0();
    return 0;
}


