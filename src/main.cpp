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


// 3-5-5 - learnRand1; loop=4 learn error=0.318215 test=64.275% time=20349s
// 3-6-5 - momentum2;  loop=3 learn error=0.294996 test=67.595% time=16767s
// 3-7-5 - momentum2;  loop=8 learn error=0.266593 test=73.475% time=31857s
// 3-8-5 -
void fromStart() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
    }

    const unsigned long long rndSeed = std::random_device()();
    stdOut << "rndSeed=" << rndSeed << endl;
    FRnd rnd(rndSeed);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-8;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << "learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.toUniqueWeights(0);
    stdOut << "learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.randWeights(rnd,-1.0,+1.0);
    nn.save( "nndef_out.txt" );

    if( true )
    {
        NNNet orgNN;
        orgNN.read("nndef.txt" );
        orgNN.toUniqueWeights(1);
        orgNN.setMinWeights(-30);
        orgNN.setMaxWeights(+30);
        CNNData subLearn = learn.rndSelect(500,rnd());
        nnftyp e = nn.error( subLearn, bigPenal);
        for( nnityp loop=1 ; loop <= 50000 ; loop ++ ) {
            NNNet tmpNN = orgNN;
            tmpNN.randWeights(rnd,-0.1,+0.1);
            nncftyp tmpE = tmpNN.error( subLearn , bigPenal );
            if( tmpE < e || loop==1 ) {
                e = tmpE;
                nn = tmpNN;
                stdOut << "init loop=" << loop << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );
            }
        }
        stdOut << "learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    }

    TVFlt toLearn( nn.sizeWeights() , 1.0 );
//    toLearn[0] = 0;


    {
        TVInt parts = { 500,1000,1500,2000,3500,5000,10000,20000,30000,learn.size()/2};
        for( nnityp loop=0 ; loop<parts.size() ; loop++ ) {
            NNData subLearn,subTest;
            PEXCP( parts[loop]*2 <= learn.size() );
            if( parts[loop]*4 <= learn.size() ) {
                learn.split(subLearn,parts[loop],subTest,parts[loop]*3,rnd());
            } else if( parts[loop]*3 <= learn.size() ) {
                learn.split(subLearn,parts[loop],subTest,parts[loop]*2,rnd());
            } else {
                learn.split(subLearn,parts[loop],subTest,parts[loop]*1,rnd());
            }
            NNNet best = nn;
            nncftyp maxStep   =  parts[loop] < 500 ? 1E-2 : parts[loop] <= 1000 ? 5E-2 : 1E-1;
            nncftyp bigPenal2 =  bigPenal;

            nn.momentum2( subLearn , subTest, 3600*24, 30, 1E-6, maxStep, 1E-9, 0.80, toLearn, bigPenal2, 5, &best);
            nn = best;
            stdOut << "momentum2;  loop=" << loop;
            stdOut << " part=" << parts[loop];
            stdOut << " learn error=" << nn.error( learn , bigPenal );
            stdOut << " test=" << classify( test ,nn ) << "%";
            stdOut << " time=" << (time(NULL)-start) << "s" << endl;
            nn.save( "nndef_out.txt" );
        }
        bigPenal /= 5;
        if( bigPenal < 1E-12 ) {
            bigPenal = 1E-12;
        }
    }


    {
        for( nnityp loop=1 ; true ; loop++ ) {
            {
                nn.learnRand1( learn, bigPenal, 600, 0, 1E-12, 1E-6 , 1E-1 , rnd(), true, false, false );
                stdOut << "learnRand1; loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save("nndef_out.txt" );
            }

            {
                nn.momentum2( learn , CNNData(), 3000, 0, 1E-3, 1E-3, 1E-9, 0.5, toLearn, bigPenal, 5, nullptr);
                stdOut << "momentum2;  loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );
            }
        }
    }


}

void nextLearn() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
    }

    FRnd rnd(6);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-12;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    nn.toUniqueWeights( 1 );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
    nn.save( "nndef_out.txt" );

    {
        TVFlt toLearn( nn.sizeWeights() , 1.0 );
        toLearn[0] = 0;

        for( nnityp loop=1 ; true ; loop++ ) {

            if( loop >= 1 )
            {
                nn.momentum2( learn , CNNData(), 3000, 0, 5E-2, 5E-2, 1E-12, 0.995, toLearn, bigPenal, 5, nullptr);
                stdOut << "momentum2;  loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save( "nndef_out.txt" );
            }

            if( loop >= 1 )
            {
                //nn.annealing(learn,bigPenal,1800,1E-6,1E-3,0.4,rnd());
                nn.learnRand1(learn,bigPenal,600,0,1E-9,1E-6,1E-2,rnd(),true,false,false);
                stdOut << "learnRand1; loop=" << loop;
                stdOut << " learn error=" << nn.error( learn , bigPenal );
                stdOut << " test=" << classify( test ,nn ) << "%";
                stdOut << " time=" << (time(NULL)-start) << "s" << endl;
                nn.save("nndef_out.txt" );
            }

        }
    }


}


void experiment0() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
    }

    const unsigned long long rndSeed = std::random_device()();
    stdOut << "rndSeed=" << rndSeed << endl;
    FRnd rnd(rndSeed);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-7;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    {
        nn.toUniqueWeights(0);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    {
        nn.randWeights(rnd,-0.1,+0.1);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

//    {
//        nn.randIdxW(rnd,-1,+1);
//        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
//        nn.save( "nndef_out.txt" );
//    }

//    for( nnityp loop=1 ; nn.forceIdx( learn , bigPenal , 1 , true , rnd() ) ; loop++ ) {
//        stdOut << "forceIdx;   loop=" << loop;
//        stdOut << " learn error=" << nn.error( learn , bigPenal );
//        stdOut << " test=" << classify( test ,nn ) << "%";
//        stdOut << " time=" << (time(NULL)-start) << "s" << endl;
//        nn.save("nndef_out.txt" );
//    }

//    {
//        nn.toUniqueWeights(0);
//        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
//        nn.save( "nndef_out.txt" );
//    }

    for( nnityp loop=1 ; true ; loop++ ) {
        {
            nn.momentum2( learn , CNNData(), 3000, 0, 1E-3, 1E-2, 1E-12, 0.5, CTVFlt(), bigPenal, 5, nullptr);
            stdOut << "momentum2;  loop=" << loop;
            stdOut << " learn error=" << nn.error( learn , bigPenal );
            stdOut << " test=" << classify( test ,nn ) << "%";
            stdOut << " time=" << (time(NULL)-start) << "s" << endl;
            nn.save( "nndef_out.txt" );
        }
        {
            nn.learnRand1( learn, bigPenal, 600, 0, 1E-12, 1E-6 , 1E-1 , rnd(), true, false, false );
            stdOut << "learnRand1; loop=" << loop;
            stdOut << " learn error=" << nn.error( learn , bigPenal );
            stdOut << " test=" << classify( test ,nn ) << "%";
            stdOut << " time=" << (time(NULL)-start) << "s" << endl;
            nn.save("nndef_out.txt" );
        }
    }

}

void experiment1() {
    QTextStream stdOut(stdout);

    NNData learn, test;
    {
        FRnd rnd(1);
        CNNData data = NNData::mkData(false,0,4,0,3,5,"/home/m/tmp/test_data.csv",",");
        data.split( learn , 80000 , test, data.size()-80000, rnd() );
    }

    const unsigned long long rndSeed = std::random_device()();
    stdOut << "rndSeed=" << rndSeed << endl;
    FRnd rnd(rndSeed);

    const time_t start = time(NULL);
    nnftyp bigPenal = 1E-7;

    NNNet nn;
    nn.read("nndef.txt" );
    stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;

    {
        nn.toUniqueWeights(0);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    {
        nn.randWeights(rnd,-0.1,+0.1);
        stdOut << " learn error=" << nn.error( learn , bigPenal ) << " test=" << classify( test ,nn ) << "% time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

    {
        NNData subLearn,subTest;
        learn.split( subLearn, 70000 , subTest , 10000 , rnd() );
        NNNet bestNN = nn;
        nn.momentum2(subLearn , subTest, 48*3600, 200, 1E-3, 1E-2, 1E-9, 0.90, CTVFlt(), bigPenal, 5, &bestNN);
        nn = bestNN;
        stdOut << "momentum2;  loop=" << 1;
        stdOut << " learn error=" << nn.error( learn , bigPenal );
        stdOut << " test=" << classify( test ,nn ) << "%";
        stdOut << " time=" << (time(NULL)-start) << "s" << endl;
        nn.save( "nndef_out.txt" );
    }

}

int main(int argc, char *argv[])
{
    experiment1();
    return 0;
}


